import random
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

DATA_ROOT = Path(
        ""
    )

def collect_samples(root):
        samples = []  # (image_path, label, patient_id)
        print(f"[START] root = {root.resolve()}")
        for patient_dir in sorted(root.iterdir()):
            print(f"[CHECK PATIENT] {patient_dir}")
            if not patient_dir.is_dir():
                print("  -> skip: not a directory")
                continue
            patient_id = patient_dir.name
            print(f"[PATIENT] {patient_id}")
            for mayo_dir in sorted(patient_dir.iterdir()):
                print(f"  [CHECK MAYO] {mayo_dir.name}")
                if not mayo_dir.is_dir():
                    print("    -> skip: not a directory")
                    continue
                if not mayo_dir.name.startswith("Mayo "):
                    print("    -> skip: not Mayo folder")
                    continue
                label = int(mayo_dir.name.split()[-1])  # "Mayo 2" -> 2
                print(f"    [MAYO] {mayo_dir.name}, label = {label}")
                for img_path in mayo_dir.iterdir():
                    if img_path.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"}:
                        samples.append((str(img_path), label, patient_id))
                        print(f"      [IMAGE] {img_path.name}")
        print(f"[DONE] total samples = {len(samples)}")
        return samples

class LimucDataset(Dataset):
        def __init__(self, samples, transform=None):
            self.samples = samples
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        
def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    samples = collect_samples(DATA_ROOT)

    patient_ids = sorted({pid for _, _, pid in samples})
    random.seed(42)
    random.shuffle(patient_ids)
    split_idx = int(len(patient_ids) * 0.8)
    train_ids = set(patient_ids[:split_idx])
    val_ids = set(patient_ids[split_idx:])

    train_samples = [
        (p, l)
        for p, l, pid in samples
        if pid in train_ids
    ]

    val_samples = [
        (p, l)
        for p, l, pid in samples
        if pid in val_ids
    ]

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = LimucDataset(train_samples, transform=transform)
    val_dataset = LimucDataset(val_samples, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 1. load pretrained model
    model = resnet18(
        weights = ResNet18_Weights.DEFAULT
    )

    # 2. Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # 3. Replace classifier
    num_features = model.fc.in_features   # 512

    model.fc = nn.Linear(
        num_features,
        4
    )

    model = model.to(device)

    # 4. Loss
    criterion = nn.CrossEntropyLoss()

    # 5. Optimizer
    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=1e-3
    )

    # 6. Training
    for epoch in range(5):

        model.train()

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(images)

            loss = criterion(
                output,
                labels
            )

            loss.backward()

            optimizer.step()

    # 7. Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            preds = output.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch + 1}: val_acc = {val_acc:.4f}")

if __name__ == "__main__":
    main()