import torch


def main() -> None:
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("MPS available:", torch.backends.mps.is_available())

    # Prefer MPS, then CUDA, otherwise CPU.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Selected device:", device)

    # Simple forward pass to confirm basic tensor ops on the chosen device.
    torch.set_grad_enabled(False)
    model = torch.nn.Linear(10, 2).to(device)
    x = torch.randn(4, 10, device=device)
    y = model(x)
    print("Linear forward OK. Output shape:", tuple(y.shape))

    try:
        import ultralytics
    except ImportError:
        print("Ultralytics not installed.")
        return

    checks_result = ultralytics.checks()
    if checks_result:
        print(checks_result)


if __name__ == "__main__":
    main()
