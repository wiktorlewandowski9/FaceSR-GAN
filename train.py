import torch
from PIL import Image
import os
import wandb
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast

from models.generator_v2 import Generator
from models.discriminator import Discriminator

#~~~~~~~~~~~~~~~~~ HYPERPARAMETERS ~~~~~~~~~~~~~~~~~~~~~~~~ 
BATCH_SIZE = 128
TRAIN_PATH = 'dataset_preparation/train'
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 10
LR_G = 0.0001
LR_D = 0.00001

#~~~~~~~~~~~~~~~~~ DATASET ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder):
        self.low_res_folder = os.path.join(root_folder, "low_res")
        self.high_res_folder = os.path.join(root_folder, "high_res")
        
        self.low_res_paths = [os.path.join(self.low_res_folder, f) for f in os.listdir(self.low_res_folder)]
        self.high_res_paths = [os.path.join(self.high_res_folder, f) for f in os.listdir(self.high_res_folder)]
        
        assert len(self.low_res_paths) == len(self.high_res_paths), "Mismatch in number of low-res and high-res images"
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        try:
            lr_image = self.transform(Image.open(self.low_res_paths[idx]).convert("RGB"))
            hr_image = self.transform(Image.open(self.high_res_paths[idx]).convert("RGB"))
        except Exception as e:
            raise RuntimeError(f"Error loading image at index {idx}: {e}")
        return lr_image, hr_image

    def __len__(self):
        return len(self.low_res_paths)

#~~~~~~~~~~~~~~~~~ TRAINING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~	  

def training_loop(generator, discriminator, optimizer_g, optimizer_d, criterion_d, train_dataloader, num_epochs):
    for epoch in range(num_epochs):
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for i, (lr, hr) in enumerate(train_dataloader):
                lr = lr.to(DEVICE)
                hr = hr.to(DEVICE)

                optimizer_g.zero_grad()
                optimizer_d.zero_grad()

                with autocast(device_type='cuda:1'):
                    fake_hr = generator(lr)

                    real_out = discriminator(hr)
                    fake_out = discriminator(fake_hr.detach())

                    real_labels = torch.ones_like(real_out, device=DEVICE)
                    fake_labels = torch.zeros_like(fake_out, device=DEVICE)

                    loss_real = criterion_d(real_out, real_labels)
                    loss_fake = criterion_d(fake_out, fake_labels)
                    loss_d = loss_real + loss_fake

                loss_d.backward()
                optimizer_d.step()

                with autocast(device_type='cuda:1'):
                    fake_out = discriminator(fake_hr)
                    real_labels_for_g = torch.ones_like(fake_out, device=DEVICE)

                    loss_g = criterion_d(fake_out, real_labels_for_g)

                loss_g.backward()
                optimizer_g.step()

                wandb.log({
                    "Loss/Generator": loss_g.item(),
                    "Loss/Discriminator": loss_d.item(),
                })

                pbar.update(1)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss G: {loss_g.item():.4f}, Train Loss D: {loss_d.item():.4f}")
        
        if (epoch + 1) % 5 == 0:
            torch.save(generator.state_dict(), f"trained_models/generator-e{epoch+1}.pth")

if __name__ == "__main__":
    wandb.init(
        project="FaceSR-GAN",
        config={
            "batch_size": BATCH_SIZE,
            "lr_g": LR_G,
            "lr_d": LR_D,
            "epochs": NUM_EPOCHS
        }
    )
        
    train_dataloader = DataLoader(ImageDataset(TRAIN_PATH), batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    model_g = Generator().to(DEVICE)
    model_d = Discriminator().to(DEVICE)
    
    model_g_optim = torch.optim.AdamW(model_g.parameters(), lr=LR_G, weight_decay=1e-4)
    model_d_optim = torch.optim.AdamW(model_d.parameters(), lr=LR_D, weight_decay=1e-4)
    
    criterion_g = torch.nn.CrossEntropyLoss()
    criterion_d = torch.nn.BCEWithLogitsLoss()
    
    training_loop(model_g, model_d, model_g_optim, model_d_optim, criterion_d, train_dataloader, NUM_EPOCHS)
    wandb.finish()

# ~~~~~~~~~~~~~~~~~~~~~~~~~ TESTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def test_dataloaders():
    train_dataset = ImageDataset(TRAIN_PATH)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    for lr, hr in train_dataloader:
        assert lr.shape == (BATCH_SIZE, 3, 64, 64), f"Low-res image shape mismatch: {lr.shape}"
        assert hr.shape == (BATCH_SIZE, 3, 256, 256), f"High-res image shape mismatch: {hr.shape}"
        break
