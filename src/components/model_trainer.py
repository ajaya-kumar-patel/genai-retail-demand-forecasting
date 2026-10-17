import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, input_path=None, model_dir=None, validation_days=28,):
        self.project_root = Path(__file__).resolve().parents[2]

        self.input_path = (
            Path(input_path)
            if input_path is not None
            else self.project_root
            / "data" / "processed" / "features.parquet")

        self.model_dir = (
            Path(model_dir)
            if model_dir is not None
            else self.project_root / "models"
        )

        self.validation_days = validation_days

        if validation_days < 1:
            raise ValueError("validation_days must be positive.")

    def load_data(self):
        """Load engineered features."""

        if not self.input_path.is_file():
            raise FileNotFoundError(f"Feature dataset not found: {self.input_path}")

        df = pd.read_parquet(self.input_path)

        if df.empty:
            raise ValueError("Feature dataset is empty.")

        required = {"date", "sales", "item_id"}

        missing = required - set(df.columns)

        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        df["date"] = pd.to_datetime(df["date"], errors="raise")

        if df["sales"].isna().any():
            raise ValueError("Target sales contains missing values.")

        if not np.isfinite(df["sales"]).all():
            raise ValueError("Target sales contains non-finite values.")

        if (df["sales"] < 0).any():
            raise ValueError("Target sales cannot be negative.")

        if df.duplicated(["item_id", "date"]).any():
            raise ValueError("Duplicate item-store-date records detected.")

        return df.sort_values("date").reset_index(drop=True)

    def split_data(self, df):
        unique_dates = np.sort(df["date"].unique())

        if len(unique_dates) <= self.validation_days:
            raise ValueError(
                "Not enough unique dates for the requested "
                "validation window."
            )

        validation_start = pd.Timestamp(unique_dates[-self.validation_days])

        train_df = df[df["date"] < validation_start].copy()
        val_df = df[df["date"] >= validation_start].copy()

        if train_df.empty or val_df.empty:
            raise ValueError("Chronological split produced an empty partition.")

        if train_df["date"].max() >= val_df["date"].min():
            raise ValueError("Training and validation dates overlap.")

        logger.info("Training rows: %s | Validation rows: %s",
                    f"{len(train_df):,}", f"{len(val_df):,}")

        logger.info("Training period: %s to %s",
                    train_df["date"].min(), train_df["date"].max())

        logger.info("Validation period: %s to %s",
                    val_df["date"].min(), val_df["date"].max())

        return train_df, val_df

    def prepare_features(self, train_df, val_df):
        feature_cols = [
                        # Product information
                        "item_id", "dept_id", "cat_id",

                        # Calendar features
                        "day_of_week", "day_of_month", "month", "year", "week_of_year", "is_weekend",

                        # Historical sales
                        "sales_lag_7", "sales_lag_14", "sales_lag_28", "sales_rolling_mean_7", "sales_rolling_mean_28",

                        # Retail information
                        "snap_CA", "sell_price", "price_change_pct",

                        # Event information
                        "event_name_1", "event_type_1",
                        "event_name_2", "event_type_2",
                    ]

        categorical_cols = [
                            "item_id", "dept_id", "cat_id",
                            "event_name_1", "event_type_1",
                            "event_name_2", "event_type_2",
                        ]

        numerical_cols = [col for col in feature_cols
                          if col not in categorical_cols]


        missing_features = set(feature_cols) - set(train_df.columns)

        if missing_features:
            raise ValueError(f"Missing required features: {sorted(missing_features)}")

        X_train = train_df[feature_cols].copy()
        y_train = train_df["sales"]

        X_val = val_df[feature_cols].copy()
        y_val = val_df["sales"]

        return X_train, y_train, X_val, y_val

    def build_pipeline(self, X_train):
        """Build preprocessing and XGBoost training pipeline."""

        categorical_cols = [
            "item_id", "dept_id", "cat_id",
            "event_name_1", "event_type_1",
            "event_name_2", "event_type_2",
        ]

        categorical_cols = [col for col in categorical_cols
                            if col in X_train.columns]

        numerical_cols = [col for col in X_train.columns
                          if col not in categorical_cols
                        ]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), numerical_cols),
                ("cat", Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
                        ]),
                        categorical_cols,
                        )
                ],
                remainder="drop")

        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50,
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        return pipeline

    def train(self, X_train, y_train, X_val, y_val):
        """Train XGBoost with chronological validation."""

        pipeline = self.build_pipeline(X_train)

        # Fit preprocessing only on the training dataset.
        X_train_processed = pipeline.named_steps["preprocessor"].fit_transform(X_train)

        X_val_processed = pipeline.named_steps["preprocessor"].transform(X_val)

        logger.info("Training matrix shape: %s", X_train_processed.shape)
        logger.info("Validation matrix shape: %s", X_val_processed.shape)

        # Train XGBoost using the validation set for early stopping.
        pipeline.named_steps["model"].fit(X_train_processed, y_train,
                                          eval_set=[(X_val_processed, y_val)],
                                          verbose=False)

        return pipeline

    def save_artifacts(self, pipeline):
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Save preprocessing + trained model together.
        pipeline_path = self.model_dir / "xgboost_pipeline.joblib"
        joblib.dump(pipeline, pipeline_path)

        # Save the native XGBoost model separately.
        model_path = self.model_dir / "xgboost_model.json"
        pipeline.named_steps["model"].save_model(str(model_path))

        logger.info("Saved complete pipeline to %s", pipeline_path)
        logger.info("Saved XGBoost model to %s", model_path)

        return pipeline_path

    def initiate_model_training(self):
        df = self.load_data()

        train_df, val_df = self.split_data(df)

        X_train, y_train, X_val, y_val = self.prepare_features(train_df, val_df)

        pipeline = self.train(X_train, y_train, X_val, y_val)

        model_path = self.save_artifacts(pipeline)

        logger.info("Model training completed successfully.")

        return {"model_path": str(model_path)}
