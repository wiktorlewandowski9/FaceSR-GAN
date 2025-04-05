import os
from glob import glob
from PIL import Image
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

from models.generator import Generator
from models.discriminator import Discriminator

# ----------- Dataset ------------

class FaceDataset(Dataset):
    def __init__(self, folder_path, high_res_size=128, scale_factor=4):
        self.image_paths = glob(os.path.join(folder_path, "*"))
        self.hr_transform = transforms.Compose([
            transforms.Resize((high_res_size, high_res_size)),
            transforms.ToTensor(),
        ])
        self.low_res_size = high_res_size // scale_factor
        self.lr_transform = transforms.Compose([
            transforms.Resize((self.low_res_size, self.low_res_size), interpolation=Image.BICUBIC),
            transforms.Resize((high_res_size, high_res_size), interpolation=Image.BICUBIC),  # Upscale to match HR
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        hr = self.hr_transform(img)
        lr = self.lr_transform(img)
        return lr, hr

# ----------- Losses ------------

def generator_loss(fake_pred):
    return nn.BCELoss()(fake_pred, torch.ones_like(fake_pred))

def discriminator_loss(real_pred, fake_pred):
    real_loss = nn.BCELoss()(real_pred, torch.ones_like(real_pred))
    fake_loss = nn.BCELoss()(fake_pred, torch.zeros_like(fake_pred))
    return real_loss + fake_loss

# ----------- Training Loop ------------

def train_loop(
    data_dir="creating_dataset/split_data/train",
    val_dir="creating_dataset/split_data/validate",
    epochs=40,
    batch_size=32,
    lr=0.0002,
    save_path="trained_models",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    os.makedirs(save_path, exist_ok=True)

    train_dataset = FaceDataset(data_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    val_dataset = FaceDataset(val_dir)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    gen_opt = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        g_losses, d_losses = [], []

        for lr_imgs, hr_imgs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            # ---- Train Discriminator ----
            fake_imgs = generator(lr_imgs)
            real_pred = discriminator(hr_imgs)
            fake_pred = discriminator(fake_imgs.detach())

            d_loss = discriminator_loss(real_pred, fake_pred)

            disc_opt.zero_grad()
            d_loss.backward()
            disc_opt.step()

            # ---- Train Generator ----
            fake_pred = discriminator(fake_imgs)
            g_loss = generator_loss(fake_pred)

            gen_opt.zero_grad()
            g_loss.backward()
            gen_opt.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        print(f"[Epoch {epoch+1}] Generator loss: {sum(g_losses)/len(g_losses):.4f} | Discriminator loss: {sum(d_losses)/len(d_losses):.4f}")

    # Save models
    torch.save(generator.state_dict(), os.path.join(save_path, "generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(save_path, "discriminator.pth"))
    print(f"Models saved to {save_path}/")

# ----------- Run Script ------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="creating_dataset/split_data/train")
    parser.add_argument("--val_dir", type=str, default="creating_dataset/split_data/validate")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--save_path", type=str, default="trained_models")

    args = parser.parse_args()

    train_loop(
        data_dir=args.train_dir,
        val_dir=args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path
    )
