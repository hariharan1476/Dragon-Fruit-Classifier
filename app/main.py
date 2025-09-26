from fastapi import FastAPI, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil, os
import numpy as np
from huggingface_hub import hf_hub_download
import keras
from tensorflow.keras.preprocessing import image

app = FastAPI()

# Templates and static files
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/uploads", StaticFiles(directory="app/uploads"), name="uploads")

# Hugging Face model
REPO_ID = "Hari1476ee/dragon-fruit-classifier"
FILENAME = "model2.keras"
HF_TOKEN = "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"

if not os.path.exists(FILENAME):
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, use_auth_token=HF_TOKEN)
else:
    model_path = FILENAME

model = keras.saving.load_model(model_path)
class_labels = ["Defect", "Fresh", "Immature", "Infected"]

# Home page
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "name": "Hariharan Krishnamoorthy",
        "linkedin": "https://www.linkedin.com/in/hari-haran-k-08a8a1249/"
    })

# Prediction endpoint
@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    upload_folder = "app/uploads"
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = image.load_img(file_path, target_size=(150, 150))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = model.predict(img_array)
    pred_class = class_labels[np.argmax(preds)]

    return templates.TemplateResponse("result.html", {
        "request": request,
        "prediction": pred_class,
        "filename": file.filename,
        "name": "Hariharan Krishnamoorthy",
        "linkedin": "https://www.linkedin.com/in/hari-haran-k-08a8a1249/"
    })
