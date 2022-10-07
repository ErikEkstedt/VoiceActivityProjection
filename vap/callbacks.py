import torch
import pytorch_lightning as pl
import wandb


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
