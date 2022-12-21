from argparse import ArgumentParser
from os.path import basename, join
from pathlib import Path
from pytorch_lightning import Trainer

import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets_turntaking import DialogAudioDM
from vap.callbacks import SymmetricSpeakersCallback
from vap.events import TurnTakingEvents, EventConfig
from vap.model import VapGPT, VapConfig, load_older_state_dict
from vap.zero_shot import ZeroShot
from vap.train import VAPModel, DataConfig
from vap.utils import (
    everything_deterministic,
    write_json,
    read_json,
    tensor_dict_to_json,
)


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
    parser = DataConfig.add_argparse_args(parser)
    parser, fields_added = VapConfig.add_argparse_args(parser)
    parser, fields_added = EventConfig.add_argparse_args(parser, fields_added)
    args = parser.parse_args()
    return args, {
        "model": VapConfig.args_to_conf(args),
        "event": EventConfig.args_to_conf(args),
        "data": DataConfig.args_to_conf(args),
    }


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


def get_savepath(args, configs):
    name = basename(args.checkpoint).replace(".ckpt", "")
    name += "_" + "_".join(configs["data"].datasets)
    savepath = join(ROOT, name)
    Path(savepath).mkdir(exist_ok=True, parents=True)
    print("SAVEPATH: ", savepath)
    # write_json(cfg_dict, join(savepath, "config.json"))
    return savepath


def get_phrases_eval(model):
    from vap.phrases.dataset import PhraseDataset
    from vap.utils import batch_to_device
    from tqdm import tqdm

    def get_last_frame_active(vad, channel=0):
        lasts = torch.zeros(vad.shape[0])
        for b in range(vad.shape[0]):
            w = torch.where(vad[b, :, channel] == 1)[0][-1]
            lasts[b] = w
        lasts -= 1
        return lasts.long()

    # TODO: make callback to get
    #          H   P   R
    # now:  0.96 0.81 0.29
    # fut:  0.76 0.52 0.3

    # TODO: better pad function
    def collate_fn(batch):
        waveform = []
        vad = []
        ns_max = -1
        nf_max = -1
        for b in batch:
            waveform.append(b["waveform"])
            vad.append(b["vad"])
            if b["waveform"].shape[-1] > ns_max:
                ns_max = b["waveform"].shape[-1]
            if b["vad"].shape[-2] > nf_max:
                nf_max = b["vad"].shape[-2]

        w = torch.zeros((len(waveform), 2, ns_max))
        for ii, ww in enumerate(waveform):
            w[ii, :, : ww.shape[-1]] = ww[0]

        v = torch.zeros((len(vad), nf_max, 2))
        for ii, vv in enumerate(vad):
            v[ii, : vv.shape[-2], :] = vv[0]

        return {"waveform": w, "vad": v}

    model.eval()
    model.to("cuda")
    dset = PhraseDataset()
    dloader = DataLoader(
        dset,
        batch_size=4,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn,
    )
    npad = 10
    with torch.no_grad():
        p_hold, p_pred, p_react = [], [], []
        pf_hold, pf_pred, pf_react = [], [], []
        pt_hold, pt_pred, pt_react = [], [], []
        for batch in tqdm(dloader):
            batch = batch_to_device(batch, model.device)
            out = model.shared_step(batch)
            probs = model.objective.get_probs(out["logits"])
            last_ind = get_last_frame_active(batch["vad"])
            for b in range(len(last_ind)):
                beg_sil = last_ind[b]
                ph = probs["p_now"][b, : beg_sil - npad, 0].mean().cpu()
                pp = probs["p_now"][b, beg_sil - npad : beg_sil, 0].cpu()
                pr = probs["p_now"][b, beg_sil : beg_sil + npad, 0].cpu()
                pfh = probs["p_future"][b, : beg_sil - npad, 0].mean().cpu()
                pfp = probs["p_future"][b, beg_sil - npad : beg_sil, 0].cpu()
                pfr = probs["p_future"][b, beg_sil : beg_sil + npad, 0].cpu()
                pth = probs["p_tot"][b, : beg_sil - npad, 0].mean().cpu()
                ptp = probs["p_tot"][b, beg_sil - npad : beg_sil, 0].cpu()
                ptr = probs["p_tot"][b, beg_sil : beg_sil + npad, 0].cpu()
                p_hold.append(ph)
                p_pred.append(pp)
                p_react.append(pr)
                pf_hold.append(pfh)
                pf_pred.append(pfp)
                pf_react.append(pfr)
                pt_hold.append(pth)
                pt_pred.append(ptp)
                pt_react.append(ptr)
    p_hold = torch.stack(p_hold)
    p_pred = torch.stack(p_pred)
    p_react = torch.stack(p_react)
    pf_hold = torch.stack(pf_hold)
    pf_pred = torch.stack(pf_pred)
    pf_react = torch.stack(pf_react)
    pt_hold = torch.stack(pt_hold)
    pt_pred = torch.stack(pt_pred)
    pt_react = torch.stack(pt_react)
    phm = round(1 - p_hold.mean().cpu().item(), 2)
    ppm = round(1 - p_pred.mean().cpu().item(), 2)
    prm = round(1 - p_react.mean().cpu().item(), 2)
    pfhm = round(1 - pf_hold.mean().cpu().item(), 2)
    pfpm = round(1 - pf_pred.mean().cpu().item(), 2)
    pfrm = round(1 - pf_react.mean().cpu().item(), 2)
    pthm = round(1 - pt_hold.mean().cpu().item(), 2)
    ptpm = round(1 - pt_pred.mean().cpu().item(), 2)
    ptrm = round(1 - pt_react.mean().cpu().item(), 2)
    print("         H   P   R")
    print("now: ", phm, ppm, prm)
    print("fut: ", pfhm, pfpm, pfrm)
    print("tot: ", pthm, ptpm, ptrm)
    return {
        "now": [phm, ppm, prm],
        "fut": [pfhm, pfpm, pfrm],
        "tot": [pthm, ptpm, ptrm],
    }


def evaluate() -> None:
    """Evaluate model"""

    args, configs = get_args()
    savepath = get_savepath(args, configs)
    #########################################################
    # Load model
    #########################################################
    # model = VAPModel.load_from_checkpoint(args.checkpoint)
    model = VAPModel(configs["model"], event_conf=configs["event"])
    sd = load_older_state_dict()
    model.load_state_dict(sd, strict=False)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    #########################################################
    # Load data
    #########################################################
    dconf = configs["data"]
    dm = DialogAudioDM(
        # datasets=dconf.datasets,
        datasets=["switchboard", "fisher"],
        type=dconf.type,
        audio_duration=dconf.audio_duration,
        audio_normalize=dconf.audio_normalize,
        audio_overlap=dconf.audio_overlap,
        flip_channels=dconf.flip_channels,
        flip_probability=dconf.flip_probability,
        mask_vad=dconf.mask_vad,
        mask_vad_probability=dconf.mask_vad_probability,
        # batch_size=dconf.batch_size,
        batch_size=16,
        num_workers=4,
    )
    dm.prepare_data()
    dm.setup()

    # #########################################################
    # # Threshold
    # #########################################################
    # # Find the best thresholds (S-pred, BC-pred, S/L) on the validation set
    # # threshold_path = cfg.get("thresholds", None)
    # # if threshold_path is None:
    # #     thresholds = find_threshold(
    # #         model, dm.val_dataloader(), savepath=savepath, min_thresh=MIN_THRESH
    # #     )
    # # else:
    # #     print("Loading thresholds: ", threshold_path)
    # #     thresholds = read_json(threshold_path)

    #########################################################
    # Score
    #########################################################
    trainer = Trainer(
        gpus=-1,
        deterministic=True,
        callbacks=[SymmetricSpeakersCallback()],
        precision=16,
    )
    result = trainer.test(model, dataloaders=dm.val_dataloader())[0]
    flat = {}
    for k, v in result.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k.replace('test_', '')}_{kk}"] = vv.cpu().item()
    df = pd.DataFrame([flat])
    filepath = join(savepath, "score.csv")
    df.to_csv(filepath, index=False)

    ##################################
    # Phrases
    ##################################
    phrases = get_phrases_eval(model)
    for region, (h, p, r) in phrases.items():
        df[f"{region}_hold"] = h
        df[f"{region}_pred"] = p
        df[f"{region}_reac"] = r

    df.to_csv(filepath, index=False)
    print("Saved to -> ", filepath)

    """Evaluation
    Event Acc
    - Shift-Acc, Pred-Shift-Acc

    Phrases
    --------
    * Stats:    Hold, Pred, React
        - now
        - fut
        - tot
    * Acc
        - Total
        - Correct short
        - Correct long
    """

    # dfz = pd.read_csv("test_score.csv")
    # df = pd.read_csv("test_now_fut_score.csv")
    # print("Probs")
    # print(df)
    # print("ZeroShot")
    # print(dfz)

    # from vap.utils import batch_to_device
    # batch = next(iter(dm.val_dataloader()))
    # batch = batch_to_device(batch, model.device)
    # out = model(batch["waveform"])
    # probs = model.objective.get_probs(out["logits"])

    # d = dset[0]
    # d["waveform"] = add_zero_channel(d["waveform"])

    # # result = test(model, dm.test_dataloader(), online=False)[0]
    # metrics = model.test_metric.compute()
    # metrics["loss"] = result["test_loss"]
    # metrics["threshold_pred_shift"] = thresholds["pred_shift"]
    # metrics["threshold_pred_bc"] = thresholds["pred_bc"]
    # metrics["threshold_short_long"] = thresholds["short_long"]
    #
    # #########################################################
    # # Save
    # #########################################################
    # metric_json = tensor_dict_to_json(metrics)
    # write_json(metric_json, join(savepath, "metric.json"))
    # print("Saved metrics -> ", join(savepath, "metric.json"))


if __name__ == "__main__":
    evaluate()
