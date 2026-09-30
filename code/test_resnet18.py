import torch
from torchvision.models import resnet18, ResNet18_Weights

print("=" * 60)
print("RESNET18 SETUP TEST")
print("=" * 60)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nDevice: {device}")

weights = ResNet18_Weights.DEFAULT

print("Loading pretrained ResNet18...")

model = resnet18(weights=weights)
model = model.to(device)

print("ResNet18 loaded successfully.")

# Test with one fake 224x224 RGB image
dummy = torch.randn(
    1, 3, 224, 224
).to(device)

model.eval()

with torch.no_grad():
    output = model(dummy)

print(f"Input shape : {dummy.shape}")
print(f"Output shape: {output.shape}")

print("\n" + "=" * 60)
print("TEST PASSED")
print("=" * 60)