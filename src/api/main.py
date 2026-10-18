
from contextlib import asynccontextmanager
from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipelines.prediction_pipeline import PredictionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_pipeline.joblib"

pipeline = None

class PredictionRequest(BaseModel):
    item_id: str
    dept_id: str
    cat_id: str

    day_of_week: int = Field(ge=0, le=6)
    day_of_month: int = Field(ge=1, le=31)
    month: int = Field(ge=1, le=12)
    year: int = Field(ge=2011, le=2035)
    week_of_year: int = Field(ge=1, le=53)
    is_weekend: int = Field(ge=0, le=1)

    sales_lag_7: float = Field(ge=0)
    sales_lag_14: float = Field(ge=0)
    sales_lag_28: float = Field(ge=0)
    sales_rolling_mean_7: float = Field(ge=0)
    sales_rolling_mean_28: float = Field(ge=0)

    snap_CA: int = Field(ge=0, le=1)
    sell_price: float = Field(ge=0)
    price_change_pct: float

    event_name_1: str
    event_type_1: str
    event_name_2: str
    event_type_2: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline

    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")

    pipeline = PredictionPipeline(model_path=MODEL_PATH)

    logger.info("Prediction API initialized.")
    yield
    logger.info("Prediction API shutting down.")


app = FastAPI(
    title="CA_1 Retail Demand Forecasting API",
    description="Predict item-level daily sales for the M5 CA_1 store.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "message": "CA_1 Retail Demand Forecasting API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail="Model artifact unavailable")

    return {"status": "healthy", "store_id": "CA_1"}


@app.post("/predict")
def predict(request: PredictionRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Prediction service not initialized")

    try:
        return pipeline.predict(**request.model_dump())
    except Exception:
        logger.exception("Prediction request failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
