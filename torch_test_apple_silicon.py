import torch
print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = torch.nn.Linear(10, 2).to(device)
x = torch.randn(4, 10, device=device)
y = model(x)

import ultralytics
print(ultralytics.checks())