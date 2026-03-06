import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent))
from models import CatDogScatNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
OUT_DIR      = PROJECT_ROOT / "outputs"

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 128
BATCH_SIZE = 64
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

test_ds     = datasets.ImageFolder(DATA_DIR / "test", transform=eval_tf)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = CatDogScatNet().to(DEVICE)
model.load_state_dict(torch.load(OUT_DIR / "scatnet_best.pth", map_location=DEVICE))
model.eval()

correct = total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = model(imgs)
        preds  = (torch.sigmoid(logits) >= 0.5).long().squeeze(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

print(f"ScatNet Test Accuracy: {correct/total:.4f}")
