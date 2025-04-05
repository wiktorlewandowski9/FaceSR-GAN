import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from glob import glob
from PIL import Image
from tqdm import tqdm
from models.generator import Generator
from models.discriminator import Discriminator

# ----------- Dataset ------------

class FaceDataset(Dataset):
    """
    Custom Dataset class for loading low-resolution and high-resolution facial images.
    The dataset assumes that images are located in the specified folder and pairs each low-resolution 
    image with its corresponding high-resolution image after applying transformations.
    """
    def __init__(self, folder_path, high_res_size=128, scale_factor=4):
        """
        Initializes the dataset by loading image paths and defining transformation pipelines.

        Args:
            folder_path (str): Directory containing the images.
            high_res_size (int): The size (height/width) for high-resolution images after resizing.
            scale_factor (int): The factor by which to scale down the high-resolution image to create low-res images.
        """
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
        """
        Returns the total number of images in the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves the low-res and high-res image pair for a given index.
        
        Args:
            idx (int): The index of the image to retrieve.
        
        Returns:
            tuple: A tuple containing the low-resolution and high-resolution image tensors.
        """
        img = Image.open(self.image_paths[idx]).convert("RGB")
        hr = self.hr_transform(img)
        lr = self.lr_transform(img)
        return lr, hr

# ----------- Losses ------------

def generator_loss(fake_pred):
    """
    Computes the generator loss for the GAN. This loss encourages the generator to produce fake images
    that the discriminator classifies as real.

    Args:
        fake_pred (Tensor): The discriminator's prediction for the fake images.

    Returns:
        Tensor: The loss for the generator.
    """
    return nn.BCELoss()(fake_pred, torch.ones_like(fake_pred))

def discriminator_loss(real_pred, fake_pred):
    """
    Computes the discriminator loss for the GAN. This loss encourages the discriminator to correctly
    classify real images as real and fake images as fake.

    Args:
        real_pred (Tensor): The discriminator's prediction for real images.
        fake_pred (Tensor): The discriminator's prediction for fake images.

    Returns:
        Tensor: The total loss for the discriminator (sum of real and fake losses).
    """
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
    """
    Main function that runs the GAN training loop, training the generator and discriminator models.

    Args:
        data_dir (str): Directory for training data.
        val_dir (str): Directory for validation data.
        epochs (int): Number of epochs to train.
        batch_size (int): Number of samples per batch.
        lr (float): Learning rate for the optimizer.
        save_path (str): Directory where trained models will be saved.
        device (str): Device to use for training, either "cuda" (GPU) or "cpu".
    """
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
    """
    Argument parser to handle command-line inputs for training the GAN model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="creating_dataset/split_data/train", help="Directory containing training images.")
    parser.add_argument("--val_dir", type=str, default="creating_dataset/split_data/validate", help="Directory containing validation images.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs to train the model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size used in training.")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizers.")
    parser.add_argument("--save_path", type=str, default="trained_models", help="Where to save the trained models.")

    # Parse the arguments and pass them to the training function
    args = parser.parse_args()

    train_loop(
        data_dir=args.train_dir,
        val_dir=args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path
    )
