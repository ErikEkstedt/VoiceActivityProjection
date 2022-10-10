from omegaconf import DictConfig, OmegaConf
from os.path import basename, join
from pathlib import Path
from pytorch_lightning import Trainer
import hydra
import torch

from vap.model import VAPModel
from vap.callbacks import SymmetricSpeakersCallback
from vap.utils import (
    everything_deterministic,
    write_json,
    read_json,
    tensor_dict_to_json,
)
from datasets_turntaking import DialogAudioDM

everything_deterministic()

MIN_THRESH = 0.01  # Minimum `threshold` limit for S/L, S-pred, BC-pred
ROOT = "runs_evaluation"


def get_curves(preds, target, pos_label=1, thresholds=None, EPS=1e-6):
    """
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)

    """

    if thresholds is None:
        thresholds = torch.linspace(0, 1, steps=101)

    if pos_label == 0:
        raise NotImplemented("Have not done this")

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
    dloader: torch.utils.data.DataLoader,
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
    _trainer = Trainer(
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


@hydra.main(config_path="conf", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate model"""
    cfg_dict = OmegaConf.to_object(cfg)
    cfg_dict = dict(cfg_dict)

    # Create savepaths
    name = basename(cfg.checkpoint_path).replace(".ckpt", "")
    name += "_" + "_".join(cfg.data.datasets)
    savepath = join(cfg_dict.get("savepath", ROOT), name)
    print("SAVEPATH: ", savepath)
    Path(savepath).mkdir(exist_ok=True, parents=True)
    write_json(cfg_dict, join(savepath, "config.json"))

    #########################################################
    # Load model
    #########################################################
    model = VAPModel.load_from_checkpoint(cfg.checkpoint_path)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    #########################################################
    # Load data
    #########################################################
    print("Datasets: ", cfg.data.datasets)
    print("Model stereo: ", model.stereo)
    print("Model frame_hz: ", model.frame_hz)
    print("Duration: ", cfg.data.audio_duration)
    print("VAD-hist: ", cfg.data.vad_history)
    print("Overlap: ", cfg.data.audio_overlap)
    print("Sample rate: ", cfg.data.sample_rate)
    print("Num Workers: ", cfg.data.num_workers)
    print("Batch size: ", cfg.data.batch_size)
    print(cfg.data.datasets)
    dm = DialogAudioDM(
        datasets=cfg.data.datasets,
        type=cfg.data.type,
        audio_duration=cfg.data.audio_duration,
        audio_normalize=cfg.data.audio_normalize,
        audio_overlap=cfg.data.audio_overlap,
        audio_mono=not model.stereo,
        sample_rate=cfg.data.sample_rate,
        vad_hz=model.frame_hz,
        vad_horizon=model.VAP.horizon,
        vad_history=False if model.stereo else cfg.data.vad_history,
        vad_history_times=cfg.data.vad_history_times,
        flip_channels=False,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    print(dm)
    dm.prepare_data()
    dm.setup(None)

    #########################################################
    # Threshold
    #########################################################
    # Find the best thresholds (S-pred, BC-pred, S/L) on the validation set
    threshold_path = cfg.get("thresholds", None)
    if threshold_path is None:
        thresholds = find_threshold(
            model, dm.val_dataloader(), savepath=savepath, min_thresh=MIN_THRESH
        )
    else:
        print("Loading thresholds: ", threshold_path)
        thresholds = read_json(threshold_path)

    #########################################################
    # Score
    #########################################################
    print("#" * 60)
    print("Final Score (test-set)...")
    print("#" * 60)
    model.test_metric = model.init_metric(
        threshold_pred_shift=thresholds.get("pred_shift", 0.5),
        threshold_short_long=thresholds.get("short_long", 0.5),
        threshold_bc_pred=thresholds.get("pred_bc", 0.5),
    )
    _trainer = Trainer(
        gpus=-1,
        deterministic=True,
        callbacks=[SymmetricSpeakersCallback()],
    )
    result = _trainer.test(model, dataloaders=dm.test_dataloader())[0]
    # result = test(model, dm.test_dataloader(), online=False)[0]
    metrics = model.test_metric.compute()
    metrics["loss"] = result["test_loss"]
    metrics["threshold_pred_shift"] = thresholds["pred_shift"]
    metrics["threshold_pred_bc"] = thresholds["pred_bc"]
    metrics["threshold_short_long"] = thresholds["short_long"]

    #########################################################
    # Save
    #########################################################
    metric_json = tensor_dict_to_json(metrics)
    write_json(metric_json, join(savepath, "metric.json"))
    print("Saved metrics -> ", join(savepath, "metric.json"))


if __name__ == "__main__":
    evaluate()
