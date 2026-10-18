import logging
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """Predict sales for a single input row."""
    def __init__(self, model_path=None):
        project_root = Path(__file__).resolve().parents[2]

        self.model_path = (
            Path(model_path)
            if model_path
            else project_root / "models" / "xgboost_pipeline.joblib"
        )

        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        logger.info("Loading model from %s", self.model_path)
        self.model = joblib.load(self.model_path)
        logger.info("Model loaded successfully.")

    def predict(
        self,
        item_id,
        dept_id,
        cat_id,
        day_of_week,
        day_of_month,
        month,
        year,
        week_of_year,
        is_weekend,
        sales_lag_7,
        sales_lag_14,
        sales_lag_28,
        sales_rolling_mean_7,
        sales_rolling_mean_28,
        snap_CA,
        sell_price,
        price_change_pct,
        event_name_1,
        event_type_1,
        event_name_2,
        event_type_2,
    ):
        try:
            input_data = {
                "item_id": [item_id],
                "dept_id": [dept_id],
                "cat_id": [cat_id],
                "day_of_week": [day_of_week],
                "day_of_month": [day_of_month],
                "month": [month],
                "year": [year],
                "week_of_year": [week_of_year],
                "is_weekend": [is_weekend],
                "sales_lag_7": [sales_lag_7],
                "sales_lag_14": [sales_lag_14],
                "sales_lag_28": [sales_lag_28],
                "sales_rolling_mean_7": [sales_rolling_mean_7],
                "sales_rolling_mean_28": [sales_rolling_mean_28],
                "snap_CA": [snap_CA],
                "sell_price": [sell_price],
                "price_change_pct": [price_change_pct],
                "event_name_1": [event_name_1],
                "event_type_1": [event_type_1],
                "event_name_2": [event_name_2],
                "event_type_2": [event_type_2],
            }

            input_df = pd.DataFrame(input_data)

            prediction = self.model.predict(input_df)

            predicted_sales = max(0.0, float(prediction[0]))

            logger.info("Sales prediction completed successfully.")

            return {
                "item_id": item_id,
                "predicted_sales": predicted_sales,
            }

        except Exception:
            logger.exception("Sales prediction failed.")
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    pipeline = PredictionPipeline()

    result = pipeline.predict(
        item_id="HOBBIES_1_001",
        dept_id="HOBBIES_1",
        cat_id="HOBBIES",
        day_of_week=3,
        day_of_month=15,
        month=11,
        year=2016,
        week_of_year=46,
        is_weekend=0,
        sales_lag_7=4,
        sales_lag_14=6,
        sales_lag_28=5,
        sales_rolling_mean_7=4.7,
        sales_rolling_mean_28=5.2,
        snap_CA=0,
        sell_price=9.97,
        price_change_pct=0.0,
        event_name_1="None",
        event_type_1="None",
        event_name_2="None",
        event_type_2="None",
    )

    print(result)
