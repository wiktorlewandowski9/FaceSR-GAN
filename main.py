from fastapi import FastAPI
from pydantic import BaseModel
from base64 import b64decode, b64encode
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from inference.predict import predict

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
@app.post("/process_image")
async def process_image(request: ImageRequest):
    try:
        image_data = b64decode(request.image.split(",")[1])
        processed_image = await predict(image_data)
        
        res = f"data:image/png;base64,{b64encode(processed_image).decode('utf-8')}"
        return {"processed_image": res}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)