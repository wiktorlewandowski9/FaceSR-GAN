from fastapi import FastAPI, Request
from pydantic import BaseModel
import base64
import uvicorn
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

class ImageRequest(BaseModel):
    image: str

# Serve the main HTML page
@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("static/index.html", "r") as file:
        html_content = file.read()
    return html_content

# Handle image processing requests
@app.post("/predict")
async def predict(request: ImageRequest):
    image_data = request.image.split(",")[1]

    # TODO: Add model prediction code here
    # For now, we will just return the input image

    processed_image = f"data:image/png;base64,{image_data}"
    return {"processed_image": processed_image}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)