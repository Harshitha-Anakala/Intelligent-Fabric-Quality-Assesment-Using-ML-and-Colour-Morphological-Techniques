import os
import cv2
import numpy as np
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# ===============================
# 1. Dataset Loader with Synthetic Shadows
# ===============================
class ShadowDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.files = glob(os.path.join(folder, "*.jpg")) + \
                     glob(os.path.join(folder, "*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def add_synthetic_shadow(self, img):
        h, w, _ = img.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        center = (np.random.randint(0, w), np.random.randint(0, h))
        axes = (np.random.randint(50, 150), np.random.randint(30, 100))
        angle = np.random.randint(0, 180)

        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)

        shadow = img.copy()
        intensity = np.random.uniform(0.4, 0.7)
        shadow[mask == 255] = (shadow[mask == 255] * intensity).astype(np.uint8)

        return shadow, img

    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))

        shadow_img, clean_img = self.add_synthetic_shadow(img)

        if self.transform:
            shadow_img = self.transform(shadow_img)
            clean_img = self.transform(clean_img)

        return shadow_img, clean_img


# ===============================
# 2. Tiny CNN for Shadow Removal
# ===============================
class ShadowRemovalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ===============================
# 3. Training Loop
# ===============================
def train_shadow_removal(loader, epochs=5, lr=1e-3, device="cpu"):
    model = ShadowRemovalCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(loader)

        for shadow, clean in loop:
            shadow, clean = shadow.to(device), clean.to(device)

            optimizer.zero_grad()
            output = model(shadow)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{epochs}] Loss {loss.item():.4f}")

        print(f"Epoch {epoch+1} | Avg Loss: {total_loss / len(loader):.4f}")

    return model


# ===============================
# 4. Inference
# ===============================
def remove_shadows_from_folder(model, input_folder, output_folder, device="cpu"):
    os.makedirs(output_folder, exist_ok=True)
    files = glob(os.path.join(input_folder, "*.jpg")) + \
            glob(os.path.join(input_folder, "*.png"))

    transform = transforms.ToTensor()
    model.eval()

    for file in files:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))

        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)[0].permute(1, 2, 0).cpu().numpy()
            output = (output * 255).astype(np.uint8)

        save_path = os.path.join(output_folder, os.path.basename(file))
        cv2.imwrite(save_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

    print(f"✅ Shadow-free images saved to {output_folder}")


# ===============================
# 5. Main
# ===============================
if __name__ == "__main__":
    input_folder = r"C:\Users\harsh\OneDrive\Desktop\Fabric Defect Dataset\defect free"
    output_folder = r"C:\Users\harsh\OneDrive\Desktop\project\new"


    transform = transforms.ToTensor()
    dataset = ShadowDataset(input_folder, transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("🔄 Training shadow removal model...")
    model = train_shadow_removal(loader, epochs=5, device=device)

    print("🔄 Removing shadows...")
    remove_shadows_from_folder(model, input_folder, output_folder, device)
