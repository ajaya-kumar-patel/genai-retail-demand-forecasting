import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, input_path=None, model_path=None, output_dir=None, validation_days=28):
        self.project_root = Path(__file__).resolve().parents[2]

        self.input_path = (
            Path(input_path)
            if input_path is not None
            else self.project_root
            / "data" / "processed" / "features.parquet"
        )

        self.model_path = (
            Path(model_path)
            if model_path is not None
            else self.project_root
            / "models" / "xgboost_pipeline.joblib"
        )

        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else self.project_root
            / "models" / "evaluation"
        )

        self.validation_days = validation_days

        if validation_days < 1:
            raise ValueError("validation_days must be positive.")

        self.feature_cols = [
            # Product information
            "item_id",
            "dept_id",
            "cat_id",

            # Calendar features
            "day_of_week",
            "day_of_month",
            "month",
            "year",
            "week_of_year",
            "is_weekend",

            # Historical sales
            "sales_lag_7",
            "sales_lag_14",
            "sales_lag_28",
            "sales_rolling_mean_7",
            "sales_rolling_mean_28",

            # Retail information
            "snap_CA",
            "sell_price",
            "price_change_pct",

            # Event information
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
        ]

    def load_data(self):
        if not self.input_path.is_file():
            raise FileNotFoundError(f"Feature dataset not found: {self.input_path}")

        df = pd.read_parquet(self.input_path)

        if df.empty:
            raise ValueError("Feature dataset is empty.")

        required = {"date", "sales", "item_id"}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        missing_features = set(self.feature_cols) - set(df.columns)

        if missing_features:
            raise ValueError(f"Missing model features: {sorted(missing_features)}")

        df["date"] = pd.to_datetime(df["date"], errors="raise")

        if df["date"].isna().any():
            raise ValueError("Date column contains missing values.")

        if df["sales"].isna().any():
            raise ValueError("Target sales contains missing values.")

        if not np.isfinite(df["sales"].to_numpy(dtype=float)).all():
            raise ValueError("Target sales contains non-finite values.")

        if (df["sales"] < 0).any():
            raise ValueError("Target sales cannot be negative.")

        if df.duplicated(["item_id", "date"]).any():
            raise ValueError("Duplicate item-date records detected.")

        return df.sort_values(["item_id", "date"]).reset_index(drop=True)

    def load_model(self):

        if not self.model_path.is_file():
            raise FileNotFoundError(f"Trained pipeline not found: {self.model_path}")

        pipeline = joblib.load(self.model_path)

        if not hasattr(pipeline, "predict"):
            raise TypeError("Loaded artifact is not a prediction pipeline.")

        logger.info("Loaded model pipeline from %s", self.model_path)

        return pipeline

    def split_data(self, df):
        unique_dates = np.sort(df["date"].unique())

        if len(unique_dates) <= self.validation_days:
            raise ValueError("Not enough unique dates for the validation window.")

        validation_start = pd.Timestamp(unique_dates[-self.validation_days])

        val_df = df[df["date"] >= validation_start].copy()

        logger.info("Validation period: %s to %s", val_df["date"].min(), val_df["date"].max())

        logger.info("Validation rows: %s", f"{len(val_df):,}")

        return val_df

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if len(y_true) == 0 or len(y_true) != len(y_pred):
            raise ValueError(
                "Actual and predicted values must have equal, "
                "non-zero lengths."
            )

        if not np.isfinite(y_true).all():
            raise ValueError("Actual values contain non-finite values.")

        if not np.isfinite(y_pred).all():
            raise ValueError("Predictions contain non-finite values.")

        errors = y_true - y_pred

        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        actual_sum = float(np.sum(np.abs(y_true)))

        wape = (float(np.sum(np.abs(errors)) / actual_sum * 100)
                if actual_sum > 0
                else None
                )

        # Forecast bias: positive means overforecasting.
        bias = float(np.mean(y_pred - y_true))

        return {
            "mae": mae,
            "rmse": rmse,
            "wape_percent": wape,
            "mean_bias": bias,
            "n_samples": int(len(y_true)),
        }

    def save_artifacts(self, metrics, predictions_df):

        self.output_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = self.output_dir / "evaluation_metrics.json"
        predictions_path = self.output_dir / "validation_predictions.parquet"

        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4, allow_nan=False)

        predictions_df.to_parquet(predictions_path, index=False,)

        logger.info("Saved evaluation metrics to %s", metrics_path)
        logger.info("Saved validation predictions to %s", predictions_path)

        return {"metrics_path": str(metrics_path),
                "predictions_path": str(predictions_path)
                }

    def initiate_model_evaluation(self):
        """Run end-to-end model evaluation."""

        df = self.load_data()
        pipeline = self.load_model()

        val_df = self.split_data(df)

        X_val = val_df[self.feature_cols].copy()
        y_val = val_df["sales"].to_numpy(dtype=float)

        logger.info("Generating validation predictions...")

        predictions = np.asarray(pipeline.predict(X_val), dtype=float)

        # Keep the same row order as the validation data.
        if len(predictions) != len(val_df):
            raise ValueError("Prediction count does not match validation row count.")

        if not np.isfinite(predictions).all():
            raise ValueError("Model generated non-finite predictions.")

        metrics = self.calculate_metrics(y_val, predictions)

        metrics["model"] = "XGBRegressor"

        metrics["validation_start"] = val_df["date"].min().strftime("%Y-%m-%d")

        metrics["validation_end"] = val_df["date"].max().strftime("%Y-%m-%d")

        metrics["validation_days"] = int(val_df["date"].nunique())

        predictions_df = val_df[["item_id", "date", "sales"]].copy()

        predictions_df = predictions_df.rename(columns={"sales": "actual_sales"})

        predictions_df["predicted_sales"] = predictions
        predictions_df["absolute_error"] = np.abs(predictions_df["actual_sales"] - predictions_df["predicted_sales"])

        artifacts = self.save_artifacts(metrics, predictions_df)

        logger.info("Model evaluation completed.")
        logger.info(
            "MAE: %.4f | RMSE: %.4f | WAPE: %s | Mean Bias: %.4f",
            metrics["mae"],
            metrics["rmse"],
            (
                f'{metrics["wape_percent"]:.2f}%'
                if metrics["wape_percent"] is not None
                else "undefined"
            ),
            metrics["mean_bias"],
        )

        return {"metrics": metrics, **artifacts}
