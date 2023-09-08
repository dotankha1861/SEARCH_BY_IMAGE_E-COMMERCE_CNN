from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, File, UploadFile, Request
from predict import predict_final
from PIL import Image
import numpy as np
import uvicorn

app = FastAPI()
@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    pred = predict_final(image) 
    return {'ids':pred}

if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8080)