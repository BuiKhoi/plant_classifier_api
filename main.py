from fastapi import FastAPI, File, UploadFile

from utils.image_helper import get_image_file
from utils.model_helper import predict_image

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict/predict_class")
async def predict_class(file: UploadFile = File(...)):
    image = await get_image_file(file)

    predicted = predict_image(image)

    return {
        "image_class": predicted[0]
    }