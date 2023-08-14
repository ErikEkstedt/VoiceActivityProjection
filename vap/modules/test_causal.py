import torch
from vap.modules.lightning_module import VAPModule, VAP
from vap.utils.utils import everything_deterministic

import matplotlib.pyplot as plt

everything_deterministic()


def plot_pre_post(pre, post, label="causality", pad_samples=0, time=False):
    ax = plt.subplot()
    x1 = torch.arange(len(pre)).float()
    x2 = (1 + pad_samples + len(pre) + torch.arange(len(post))).float()

    if time:
        x1 /= 16_000
        x2 /= 16_000

    ax.plot(x1, pre.detach().cpu(), label=f"{label}: PRE", color="g", linewidth=2)
    ax.plot(x2, post.detach().cpu(), label=f"{label}: POST", color="red", linewidth=2)
    ax.legend()
    ax.set_ylabel("Gradient")
    ax.set_xlabel("Step")
    if post.sum() > 0:
        print(f"{label} causailty FAILED")
        print("post grad sum: ", post.sum())
        print("pre  grad sum: ", pre.sum())
    plt.show()


def print_verbose(is_causal, name, pre_grad, post_grad, pad_samples, device):
    print()
    print(f"Causality {name}")
    print(f"({device}) (pad_samples: {pad_samples})")
    if is_causal:
        print(f"PASS: {name} is CAUSAL")
    else:
        print(f"FAIL: {name} is NOT CAUSAL")
        print(f"Post-gradient:", round(post_grad.sum().cpu().item(), 1), "> 0")
        print(f"Pre-gradient:", round(pre_grad.sum().cpu().item(), 1))


def causality_test_gradient_VAP(
    model: VAP,
    duration: float = 10.0,
    focus_time: float = 5.0,
    pad_samples: int = 0,
    verbose: bool = False,
    plot: bool = False,
) -> bool:
    """
    Test that the gradient is zero in the future.
    1. Create a random waveform of duration `duration` and set `requires_grad=True`
    2. Extract the model output (logits)
    3. Choose a frame in the middle of the sequence
    4. Calculate the loss gradient, on that specific frame, w.r.t the input tensor
    5. There should not be any gradient information in the future part of the input tensor

    CPC model is causal but the CNN makes it not strictly causal. The CPC model will provide
    gradient information 312 samples (20ms) into the future.
    A small amount which practically does not matter.

    Arguments:
        duration:    Duration of the input waveform in seconds
        focus_time:  Time in seconds to focus and calculate the gradient from
        pad_samples: Number of samples to pad the gradient check with
        verbose:     Print information
    Returns:
        is_causal:   True if the gradient is zero in the future
    """
    model.train()

    # Generate a random waveform
    n_samples = int(model.sample_rate * duration)
    focus_sample = int(model.sample_rate * focus_time)
    focus_frame = int(model.frame_hz * focus_time)
    # 1. Waveform + gradient tracking
    x = torch.randn(2, 2, n_samples, device=model.device, requires_grad=True)

    # 2. Model output
    out = model(x)

    # 3. Gradient calculation
    loss = out["logits"][:, focus_frame, :].sum()
    loss.backward()

    # Gradient result
    g = x.grad.abs()
    pre_grad = g[..., :focus_sample].sum(0).sum(0)
    post_grad = g[..., focus_sample + 1 + pad_samples :].sum(0).sum(0)
    is_causal = post_grad.sum() == 0

    encoder_name = model.encoder.__class__.__name__
    if verbose:
        print("VAP Encoder:", encoder_name)
        print(f"({model.device}) Future total gradient:", post_grad.sum().cpu().item())
        print(f"({model.device}) Past total gradient:", pre_grad.sum().cpu().item())
        if not is_causal:
            print(
                f"Future gradient should be zero. got  {post_grad.sum().cpu().item()}"
            )

    if plot:
        plot_pre_post(
            pre_grad,
            post_grad,
            pad_samples=pad_samples,
            label=f"VAP {encoder_name}",
            time=True,
        )

    return is_causal


def causal_test_samples_to_frames(
    model,
    duration: float = 10.0,
    focus_time: float = 5.0,
    pad_samples: int = 0,
    sample_rate: int = 16_000,
    wav_channels: int = 2,
    output_key: str = "logits",
    frame_hz: int = 50,
    device="cpu",
):
    model.train()

    # 1. Waveform + gradient tracking
    n_samples = int(sample_rate * duration)
    focus_sample = int(sample_rate * focus_time)
    focus_frame = int(frame_hz * focus_time)
    x = torch.randn(2, wav_channels, n_samples, device=device, requires_grad=True)

    # 2. Model output
    y = model(x)
    if isinstance(y, dict):
        y = y[output_key]

    # 3. Gradient calculation
    focus_loss = y[:, focus_frame, :].sum()
    focus_loss.backward()

    # Gradient result
    g = x.grad.abs()
    pre_grad = g[..., :focus_sample].sum(0).sum(0)
    post_grad = g[..., focus_sample + 1 + pad_samples :].sum(0).sum(0)
    is_causal = (post_grad.sum() == 0).item()
    return is_causal, pre_grad, post_grad


if __name__ == "__main__":

    from argparse import ArgumentParser
    from vap.modules.modules import TransformerStereo
    from vap.modules.encoder import EncoderCPC
    from vap.modules.encoder_hubert import EncoderHubert

    parser = ArgumentParser()
    parser.add_argument("--encoder", type=str, default="cpc")
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--focus", type=float, default=5)
    parser.add_argument(
        "--pad_samples", type=int, default=0, help="312 sample lookahead"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--wav_channels", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if args.focus >= args.duration:
        print("Focus time must be less than duration!")
        args.focus = args.duration // 2
        print("Setting focus time to ", args.focus)

    if args.checkpoint is not None:
        model = VAPModule.load_model(args.checkpoint)
    else:
        if args.encoder.lower() == "hubert":
            enc = EncoderHubert()
        else:
            enc = EncoderCPC()
        model = VAP(enc, TransformerStereo())

    if args.cpu:
        model = model.to("cpu")
        print("CPU")
    else:
        if torch.cuda.is_available():
            model = model.to("cuda")
            print("CUDA")

    name = model.__class__.__name__
    is_causal, pre_grad, post_grad = causal_test_samples_to_frames(
        model,
        args.duration,
        args.focus,
        args.pad_samples,
        wav_channels=args.wav_channels,
        device=model.device,
    )
    print_verbose(is_causal, name, pre_grad, post_grad, args.pad_samples, model.device)
    if args.plot:
        plot_pre_post(
            pre_grad,
            post_grad,
            pad_samples=args.pad_samples,
            label=f"{name}",
            time=True,
        )
    if not is_causal and args.encoder == "cpc":
        print("CPC is strictly causal using `pad_samples` >= 311")
        is_causal, pre_grad, post_grad = causal_test_samples_to_frames(
            model,
            args.duration,
            args.focus,
            pad_samples=311,
            device=model.device,
        )
        print_verbose(is_causal, name, pre_grad, post_grad, 311, model.device)
