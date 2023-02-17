from argparse import ArgumentParser
from os.path import basename, join
from pathlib import Path
import pandas as pd

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

from vap_dataset.datamodule import VapDataModule

from vap.callbacks import SymmetricSpeakersCallback
from vap.train import VAPModel, DataConfig, OptConfig
from vap.phrases.dataset import PhrasesCallback
from vap.utils import everything_deterministic, write_json

# Delete later prolly
from vap.model import VapGPT, VapConfig
from vap.events import TurnTakingEvents, EventConfig
from vap.zero_shot import ZeroShot


everything_deterministic()

MIN_THRESH = 0.01  # Minimum `threshold` limit for S/L, S-pred, BC-pred
ROOT = "runs_evaluation"


def get_args():
    parser = ArgumentParser("VoiceActivityProjection")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.ckpt",
    )
    parser = pl.Trainer.add_argparse_args(parser)
    # parser = OptConfig.add_argparse_args(parser)
    parser = DataConfig.add_argparse_args(parser)
    parser, fields_added = VapConfig.add_argparse_args(parser)
    parser, fields_added = EventConfig.add_argparse_args(parser, fields_added)
    args = parser.parse_args()

    model_conf = VapConfig.args_to_conf(args)
    # opt_conf = OptConfig.args_to_conf(args)
    data_conf = DataConfig.args_to_conf(args)
    event_conf = EventConfig.args_to_conf(args)

    # Remove all non trainer args
    cfg_dict = vars(args)
    for k, _ in list(cfg_dict.items()):
        if (
            k.startswith("data_")
            or k.startswith("vap_")
            or k.startswith("opt_")
            or k.startswith("event_")
        ):
            cfg_dict.pop(k)

    return {
        "args": args,
        "cfg_dict": cfg_dict,
        "model": model_conf,
        "event": event_conf,
        # "opt": opt_conf,
        "data": data_conf,
    }


def get_curves(preds, target, pos_label=1, thresholds=None, EPS=1e-6):
    """
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)

    """

    if thresholds is None:
        thresholds = torch.linspace(0, 1, steps=101)

    if pos_label == 0:
        raise NotImplementedError("Have not done this")

    ba, f1 = [], []
    auc0, auc1 = [], []
    prec0, rec0 = [], []
    prec1, rec1 = [], []
    pos_label_idx = torch.where(target == 1)
    neg_label_idx = torch.where(target == 0)

    for t in thresholds:
        pred_labels = (preds >= t).float()
        correct = pred_labels == target

        # POSITIVES
        tp = correct[pos_label_idx].sum()
        n_p = (target == 1).sum()
        fn = n_p - tp
        # NEGATIVES
        tn = correct[neg_label_idx].sum()
        n_n = (target == 0).sum()
        fp = n_n - tn
        ###################################3
        # Balanced Accuracy
        ###################################3
        # TPR, TNR
        tpr = tp / n_p
        tnr = tn / n_n
        # BA
        ba_tmp = (tpr + tnr) / 2
        ba.append(ba_tmp)
        ###################################3
        # F1
        ###################################3
        precision1 = tp / (tp + fp + EPS)
        recall1 = tp / (tp + fn + EPS)
        f1_1 = 2 * precision1 * recall1 / (precision1 + recall1 + EPS)
        prec1.append(precision1)
        rec1.append(recall1)
        auc1.append(precision1 * recall1)

        precision0 = tn / (tn + fn + EPS)
        recall0 = tn / (tn + fp + EPS)
        f1_0 = 2 * precision0 * recall0 / (precision0 + recall0 + EPS)
        prec0.append(precision0)
        rec0.append(recall0)
        auc0.append(precision0 * recall0)

        f1w = (f1_0 * n_n + f1_1 * n_p) / (n_n + n_p)
        f1.append(f1w)

    return {
        "bacc": torch.stack(ba),
        "f1": torch.stack(f1),
        "prec1": torch.stack(prec1),
        "rec1": torch.stack(rec1),
        "prec0": torch.stack(prec0),
        "rec0": torch.stack(rec0),
        "auc0": torch.stack(auc0),
        "auc1": torch.stack(auc1),
        "thresholds": thresholds,
    }


def find_threshold(
    model: VAPModel,
    dloader: DataLoader,
    savepath: str,
    min_thresh: float = 0.01,
):
    """Find the best threshold using PR-curves"""

    def get_best_thresh(curves, metric, measure, min_thresh):
        ts = curves[metric]["thresholds"]
        over = min_thresh <= ts
        under = ts <= (1 - min_thresh)
        w = torch.where(torch.logical_and(over, under))
        values = curves[metric][measure][w]
        ts = ts[w]
        _, best_idx = values.max(0)
        return ts[best_idx]

    print("#" * 60)
    print("Finding Thresholds (val-set)...")
    print("#" * 60)

    # Init metric:
    model.test_metric = model.init_metric(
        bc_pred_pr_curve=True,
        shift_pred_pr_curve=True,
        long_short_pr_curve=True,
    )

    # Find Thresholds
    _trainer = pl.Trainer(
        gpus=-1,
        deterministic=True,
        callbacks=[SymmetricSpeakersCallback()],
    )
    _ = _trainer.test(model, dataloaders=dloader)

    ############################################
    predictions = {}
    if hasattr(model.test_metric, "long_short_pr"):
        predictions["long_short"] = {
            "preds": torch.cat(model.test_metric.long_short_pr.preds),
            "target": torch.cat(model.test_metric.long_short_pr.target),
        }
    if hasattr(model.test_metric, "bc_pred_pr"):
        predictions["bc_preds"] = {
            "preds": torch.cat(model.test_metric.bc_pred_pr.preds),
            "target": torch.cat(model.test_metric.bc_pred_pr.target),
        }
    if hasattr(model.test_metric, "shift_pred_pr"):
        predictions["shift_preds"] = {
            "preds": torch.cat(model.test_metric.shift_pred_pr.preds),
            "target": torch.cat(model.test_metric.shift_pred_pr.target),
        }

    ############################################
    # Curves
    curves = {}
    for metric in ["bc_preds", "long_short", "shift_preds"]:
        curves[metric] = get_curves(
            preds=predictions[metric]["preds"], target=predictions[metric]["target"]
        )

    ############################################
    # find best thresh
    bc_pred_threshold = None
    shift_pred_threshold = None
    long_short_threshold = None
    if "bc_preds" in curves:
        bc_pred_threshold = get_best_thresh(curves, "bc_preds", "f1", min_thresh)
    if "shift_preds" in curves:
        shift_pred_threshold = get_best_thresh(curves, "shift_preds", "f1", min_thresh)
    if "long_short" in curves:
        long_short_threshold = get_best_thresh(curves, "long_short", "f1", min_thresh)

    thresholds = {
        "pred_shift": shift_pred_threshold,
        "pred_bc": bc_pred_threshold,
        "short_long": long_short_threshold,
    }

    th = {k: v.item() for k, v in thresholds.items()}
    # torch.save(prediction, join(savepath, "predictions.pt"))
    write_json(th, join(savepath, "thresholds.json"))
    torch.save(curves, join(savepath, "curves.pt"))
    print("Saved Thresholds -> ", join(savepath, "thresholds.json"))
    print("Saved Curves -> ", join(savepath, "curves.pt"))
    return thresholds


def get_savepath(args, configs):
    name = basename(args.checkpoint).replace(".ckpt", "")
    # name += "_" + "_".join(configs["data"].datasets)
    savepath = join(ROOT, name)
    Path(savepath).mkdir(exist_ok=True, parents=True)
    print("SAVEPATH: ", savepath)
    # write_json(cfg_dict, join(savepath, "config.json"))
    return savepath


def evaluate() -> None:
    """Evaluate model"""

    configs = get_args()

    args = configs["args"]
    cfg_dict = configs["cfg_dict"]
    savepath = get_savepath(args, configs)

    #########################################################
    # Load model
    #########################################################
    model = VAPModel.load_from_checkpoint(args.checkpoint)

    #########################################################
    # Load data
    #########################################################
    dconf = configs["data"]
    dm = VapDataModule(
        train_path=dconf.train_path,
        val_path=dconf.val_path,
        test_path=dconf.test_path,
        horizon=2,
        batch_size=dconf.batch_size,
        num_workers=dconf.num_workers,
    )
    dm.prepare_data()
    dm.setup("test")

    # TODO: Do we still want to use zero-shot + threshold?
    #########################################################
    # Threshold
    #########################################################
    # Find the best thresholds (S-pred, BC-pred, S/L) on the validation set
    # threshold_path = cfg.get("thresholds", None)
    # if threshold_path is None:
    #     thresholds = find_threshold(
    #         model, dm.val_dataloader(), savepath=savepath, min_thresh=MIN_THRESH
    #     )
    # else:
    #     print("Loading thresholds: ", threshold_path)
    #     thresholds = read_json(threshold_path)

    #########################################################
    # Score
    #########################################################
    for pop in ["checkpoint", "seed", "gpus"]:
        cfg_dict.pop(pop)
    cfg_dict["accelerator"] = "gpu"
    cfg_dict["devices"] = 1
    cfg_dict["deterministic"] = True
    cfg_dict["strategy"] = DDPStrategy(find_unused_parameters=False)
    trainer = pl.Trainer(
        callbacks=[SymmetricSpeakersCallback(), PhrasesCallback()], **cfg_dict
    )
    result = trainer.test(model, dataloaders=dm.test_dataloader())[0]

    # fixup results
    flat = {}
    for k, v in result.items():
        new_name = k.replace("test_", "")
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{new_name}_{kk}"] = vv.cpu().item()
        else:
            flat[new_name] = v
    df = pd.DataFrame([flat])

    name = "score"
    if cfg_dict["precision"] == 16:
        name += "_fp16"
    if cfg_dict["limit_test_batches"] is not None:
        nn = cfg_dict["limit_test_batches"] * dm.batch_size
        name += f"_nb-{nn}"

    filepath = join(savepath, name + ".csv")
    df.to_csv(filepath, index=False)
    print("Saved to -> ", filepath)


if __name__ == "__main__":
    evaluate()
