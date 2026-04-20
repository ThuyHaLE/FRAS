# dashboard/main.py
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict
import pandas as pd

from models.inference_runner import predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Predict endpoint ────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    event: Dict[str, Any]   # raw JSON object from frontend

@app.post("/api/predict")
async def predict_endpoint(req: PredictRequest):
    try:
        test_df = pd.Series(req.event).to_frame().T
        submission = predict(
            test_df=test_df,
            checkpoint_path="models/checkpoints/inference.pkl",
        )
        # submission can be a DataFrame, a dict, or any serializable object
        if isinstance(submission, pd.DataFrame):
            result = submission.to_dict(orient="records")[0]
        elif hasattr(submission, "to_dict"):
            result = submission.to_dict()
        else:
            result = submission
        return {"ok": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Serve React build ───────────────────────────────────────────────────────
app.mount("/assets", StaticFiles(directory="dashboard/dist/assets"), name="assets")

@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    return FileResponse("dashboard/dist/index.html")