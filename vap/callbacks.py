import torch
import pytorch_lightning as pl
import wandb

from vap.evaluation_phrases import evaluation_phrases
from vap.phrases.dataset import PhraseDataset

import vap.transforms as VT


class PhrasesCallback(pl.Callback):
    """
    A callback to evaluate the performance of the model over the artificially create `phrases` dataset
    """

    def __init__(self, model, phrases_path="dataset_phrases/phrases.json"):
        ######################################################
        # LOAD DATASET
        ######################################################
        is_mono = not model.stereo
        self.dset = PhraseDataset(
            phrase_path=phrases_path,
            sample_rate=model.sample_rate,
            vad_hz=model.frame_hz,
            audio_mono=is_mono,
            vad=is_mono,
            vad_history=is_mono,
        )

        ######################################################
        # TRANSFORMS
        ######################################################
        self.transforms = {
            "flat_f0": VT.FlatPitch(sample_rate=model.sample_rate),
            "only_f0": VT.LowPass(sample_rate=model.sample_rate),
            "shift_f0": VT.ShiftPitch(sample_rate=model.sample_rate),
            "flat_intensity": VT.FlatIntensity(sample_rate=model.sample_rate),
            "duration_avg": None,
        }

    def on_validation_epoch_start(self, trainer, pl_module, *args, **kwargs):
        print("DEVICE: ", pl_module.device)
        stats, fig = evaluation_phrases(
            model=pl_module,
            dset=self.dset,
            transforms=self.transforms,
            save=False,
            agg_probs=False,
        )
        eot_short = stats.stats["short"]["scp"]["regular"]
        eot_long = stats.stats["long"]["eot"]["regular"]
        pl_module.log("phrases_short", eot_short, sync_dist=True)
        pl_module.log("phrases_long", eot_long, sync_dist=True)
        pl_module.logger.experiment.log(
            data={
                "phrases": wandb.Image(fig),
                "global_step": trainer.global_step,
            },
        )

        stats, fig = evaluation_phrases(
            model=pl_module,
            dset=self.dset,
            transforms=self.transforms,
            save=False,
            agg_probs=True,
        )
        eot_short = stats.stats["short"]["scp"]["regular"]
        eot_long = stats.stats["long"]["eot"]["regular"]
        pl_module.log("phrases_short_agg", eot_short, sync_dist=True)
        pl_module.log("phrases_long_agg", eot_long, sync_dist=True)
        pl_module.logger.experiment.log(
            data={
                "phrases_agg": wandb.Image(fig),
                "global_step": trainer.global_step,
            },
        )


class SymmetricSpeakersCallback(pl.Callback):
    """
    This callback "flips" the speakers such that we get a fair evaluation not dependent on the
    biased speaker-order / speaker-activity

    The audio is mono which requires no change.

    The only change we apply is to flip the channels in the VAD-tensor and get the corresponding VAD-history
    which is defined as the ratio of speaker 0 (i.e. vad_history_flipped = 1 - vad_history)
    """

    def get_symmetric_batch(self, batch):
        """Appends a flipped version of the batch-samples"""
        for k, v in batch.items():
            if k == "vad":
                flipped = torch.stack((v[..., 1], v[..., 0]), dim=-1)
            elif k == "vad_history":
                flipped = 1.0 - v
            elif k == "waveform":
                if v.shape[1] == 2:  # stereo audio
                    flipped = torch.stack((v[:, 1], v[:, 0]), dim=1)
                else:
                    continue
            else:
                flipped = v
            batch[k] = flipped
        return batch

    def on_train_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = self.get_symmetric_batch(batch)

    def on_test_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = self.get_symmetric_batch(batch)

    def on_val_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = self.get_symmetric_batch(batch)


class WandbArtifactCallback(pl.Callback):
    def upload(self, trainer):
        run = trainer.logger.experiment
        print(f"Ending run: {run.id}")
        artifact = wandb.Artifact(f"{run.id}_model", type="model")
        for path, val_loss in trainer.checkpoint_callback.best_k_models.items():
            print(f"Adding artifact: {path}")
            artifact.add_file(path)
        run.log_artifact(artifact)

    def on_train_end(self, trainer, pl_module):
        print("Training End ---------------- Custom Upload")
        self.upload(trainer)

    def on_exception(self, trainer, pl_module, exception):
        if isinstance(exception, KeyboardInterrupt):
            print("Keyboard Interruption ------- Custom Upload")
            self.upload(trainer)
