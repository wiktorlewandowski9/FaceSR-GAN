import torch

from generator import Generator
from discriminator import Discriminator

# ~~~~~~~~~~~~~~~ Loss functions ~~~~~~~~~~~~~~~~~~~
def generator_loss(discriminator_output):
    return -torch.mean(torch.log(discriminator_output))

def discriminator_loss(real_output, fake_output):
    return -torch.mean(torch.log(real_output) + torch.log(1 - fake_output))

# ~~~~~~~~~~~~~~~ Training functions ~~~~~~~~~~~~~~~

def step(generator, discriminator, generator_optimizer, discriminator_optimizer):
    pass

def training_loop(epoch:int=40, learning_rate:float=0.001, batch_size:int=32, data_path:str="dataset/"):
    for _ in range(epoch):
        step()
    pass

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":
    generator = Generator()
    discriminator = Discriminator()

    # TODO: Load dataset, train, save model
