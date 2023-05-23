import torch
import matplotlib.pyplot as plt
from einops import rearrange
from vap.model import VAPModel
from vap.utils import load_hydra_conf


def get_forward_hook(name):
    def forward_hook(self, input, output):
        global layer_out
        if isinstance(output, torch.Tensor):
            output = [output]
        layer_out.append([name, output[0]])
        # INPUT: x, src, mask
        # OUTPUT: x, self_attn_weights, cross_attn_weights
        # xin = input[0]
        print(f"Inside {name} forward: {output[0].size()}")
        # print("input size:", input[0].size())
        # print("output size:", output[0].size())
        # xout = rearrange(output[0], "b n d -> (b n) d")
        # print("output size:", output[0].data.size())
        # print("output mean:", xout.data.mean(0).shape)
        # print("output std:", xout.data.std(0).shape)

    return forward_hook


def get_backward_hook(name):
    def backward_hook(self, grad_input, grad_output):
        global layer_grad
        if isinstance(grad_output, torch.Tensor):
            grad_output = [grad_output]
        # layer_grad[name] = grad_output[0]
        layer_grad.append([name, grad_output[0]])
        # print("Inside " + name + " backward")
        # print("grad_input size:", grad_input[0].size())
        # print("grad_output size:", grad_output[0].size())
        # xgrad = rearrange(grad_output[0], "b n d -> (b n) d")

    return backward_hook


def plot_output_and_grads(layer_out, layer_grad, figsize=(9, 12), plot=True):
    fig, ax = plt.subplots(4, 1, figsize=figsize)
    alpha = 0.6
    for name, x in layer_out:
        x = rearrange(x, "b n d -> (b n) d")
        hy, hx = torch.histogram(x, density=True, bins=100)
        tmp_a = ax[0]
        if name == "head":
            tmp_a = ax[-2]
        tmp_a.plot(
            hx[:-1].detach().cpu(),
            hy.detach().cpu(),
            alpha=alpha,
            linewidth=2,
            label=name,
        )
    for name, x in layer_grad:
        x = rearrange(x, "b n d -> (b n) d")
        hy, hx = torch.histogram(x, density=True, bins=100)
        tmp_a = ax[1]
        if name == "head":
            tmp_a = ax[-1]
        tmp_a.plot(
            hx[:-1].detach().cpu(),
            hy.detach().cpu(),
            alpha=alpha,
            linewidth=2,
            label=name,
        )
    ax[0].set_xlabel("Outputs")
    ax[0].set_title("Forward")
    ax[0].legend()
    ax[1].set_xlabel("Gradients")
    ax[1].set_title("Backward")
    ax[1].legend()
    ax[-2].set_xlabel("Outputs")
    ax[-2].set_title("Forward")
    ax[-2].legend()
    ax[-1].set_xlabel("Gradients")
    ax[-1].set_title("Backward")
    ax[-1].legend()
    plt.tight_layout()
    if plot:
        plt.pause(0.1)
    return fig, ax


def extract_label_probs():
    from tqdm import tqdm

    dm = DialogAudioDM(
        datasets=["switchboard", "fisher"],
        audio_duration=conf["data"]["audio_duration"],
        audio_mono=not conf["model"]["stereo"],
        batch_size=10,
        num_workers=0,
    )
    dm.prepare_data()
    dm.setup()

    label_count = torch.zeros(256)
    with torch.no_grad():
        for batch in tqdm(dm.train_dataloader()):
            lab = model.VAP.extract_labels(batch["vad"])
            label_count += lab.flatten(0).bincount(minlength=256).cpu()
    p_lab = label_count / label_count.sum()
    torch.save(p_lab, "p_lab.pt")
    psort, label_idx = p_lab.sort(descending=True)
    fig, ax = plt.subplots(1, 1)
    ax.plot(psort)
    plt.show()


if __name__ == "__main__":
    from datasets_turntaking import DialogAudioDM

    conf = load_hydra_conf()
    config_name = "model/vap_50hz"  # "model/vap_50hz_stereo"
    config_name = "model/vap_50hz_stereo"  # "model/vap_50hz_stereo"
    conf["model"] = load_hydra_conf(config_name=config_name)["model"]

    dm = DialogAudioDM(
        datasets=["switchboard", "fisher"],
        audio_duration=conf["data"]["audio_duration"],
        audio_mono=not conf["model"]["stereo"],
        batch_size=10,
        num_workers=0,
    )
    dm.prepare_data()
    dm.setup()

    batch = next(iter(dm.val_dataloader()))

    layer_out = []
    layer_grad = []
    model = VAPModel(conf)
    # model.net.vap_head.projection_head.bias.data = p_lab.log()
    print(model.run_name)
    if model.stereo:
        for ii, layer in enumerate(model.net.ar_channel.layers):
            name = layer.__class__.__name__ + f"_{ii}"
            layer.register_forward_hook(get_forward_hook(name))
            layer.register_full_backward_hook(get_backward_hook(name))
    for ii, layer in enumerate(model.net.ar.layers):
        name = layer.__class__.__name__ + f"_{ii}"
        layer.register_forward_hook(get_forward_hook(name))
        layer.register_full_backward_hook(get_backward_hook(name))
    _ = model.net.vap_head.register_forward_hook(get_forward_hook("head"))
    _ = model.net.vap_head.register_full_backward_hook(get_backward_hook("head"))
    out = model.shared_step(batch)
    out["loss"].backward()
    print(out["loss"])

    fig, ax = plot_output_and_grads(layer_out, layer_grad)

    for name, x in layer_out:
        print(name, x.shape)

    layer_out.keys()
