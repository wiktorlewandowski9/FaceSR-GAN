"""
This script performs super-resolution on input face images using a pre-trained GAN model.
It loads the generator model, processes the input image, and returns the enhanced image.
"""

import torch
from PIL import Image
from io import BytesIO
from torchvision import transforms
from torchvision.transforms import functional
from models.generator_v2 import Generator
from PIL import ImageFilter

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Generator()
model.load_state_dict(torch.load('trained_models/gen_ver2.pth', map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])

async def predict(image_data):
    """
    Perform super-resolution on the input image data.

    Args:
        image_data (bytes): The input image data in bytes format.

    Returns:
        bytes: The super-resolved image data in PNG format.
    """
    try:
        image = Image.open(BytesIO(image_data)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output_tensor = model(input_tensor)

        output_tensor = output_tensor.squeeze(0).cpu().detach()
        output_image = functional.to_pil_image(output_tensor.clamp(0, 1)).filter(ImageFilter.SMOOTH)

        buffer = BytesIO()
        output_image.save(buffer, format="PNG")
        ret = buffer.getvalue()
    except Exception as e:
        print(f"Error in prediction: {e}")
        ret = image_data

    return ret