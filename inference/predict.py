import torch

from PIL import Image
from io import BytesIO
from torchvision import transforms
from torchvision.transforms import functional

from generator import Generator

DEVICE =  'cuda' if torch.cuda.is_available() else 'cpu'

async def predict(image_data):
    model = Generator()
    model.to(DEVICE)
    model.eval()

    print("Model loaded to", DEVICE)

    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
    ])

    try:
        image = Image.frombytes('RGB', (16, 16), image_data)
        input = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(input)
        
        prediction_image = functional.to_pil_image(output.squeeze(0).cpu().detach())
        buffer = BytesIO()
        prediction_image.save(buffer, format="PNG")
        ret = buffer.getvalue()
    except Exception as e:
        print(e)
        ret = image_data

    return ret