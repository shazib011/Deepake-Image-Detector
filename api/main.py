from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from src.inference.predict import Predictor

app = FastAPI(title="Deepfake Detection API (Face-swap + GAN)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = Predictor()

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": predictor.models_loaded}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    result = predictor.predict_bytes(img_bytes, with_heatmap=True)
    result["filename"] = file.filename
    return result
