import torch
from vap.module import VAPModule


if __name__ == "__main__":

    path = "runs_new/VAP2/ku03sj7i/checkpoints/epoch=0-step=25.ckpt"
    module = VAPModule.load_from_checkpoint(path)
    cp = torch.load(path)
