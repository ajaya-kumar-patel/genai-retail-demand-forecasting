import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class FeatureEngineering:
    def __init__(self, input_path=None, output_path=None, lags=(7, 14, 28), rolling_windows=(7, 28)):

        self.project_root = Path(__file__).resolve().parents[2]
        self.input_path = (Path(input_path)
                           if input_path is not None
                           else self.project_root
                           /"data"/"processed"/"sales_merged.parquet")

        self.output_path = (
            Path(output_path)
            if output_path is not None
            else self.project_root
            / "data"/ "processed" / "features.parquet")

        self.lags = tuple(lags)

        self.rolling_windows = tuple(rolling_windows)

        if any(lag < 1 for lag in self.lags):
            raise ValueError("All lag values must be positive.")

        if any(window < 1 for window in self.rolling_windows):
            raise ValueError("All rolling windows must be positive.")


    def load_data(self):
        if not self.input_path.is_file():
            raise FileNotFoundError(f"Input dataset not found: {self.input_path}")

        df = pd.read_parquet(self.input_path)

        if df.empty:
            raise ValueError("Input dataset is empty.")

        required_columns = {"item_id", "date", "sales", "sell_price"}

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        logger.info("Loaded dataset with shape %s", df.shape)

        return df

    def prepare_data(self, df):
        """Validate types and establish chronological ordering."""

        df = df.copy()

        df["date"] = pd.to_datetime(df["date"], errors="raise")

        df["sales"] = pd.to_numeric(df["sales"], errors="raise")

        if df["sales"].isna().any():
            raise ValueError("Sales contains missing values.")

        if (df["sales"] < 0).any():
            raise ValueError("Sales contains negative values.")

        if df.duplicated(["item_id", "date"]).any():
            raise ValueError("Duplicate item-store-date records detected.")

        df = df.sort_values(["item_id", "date"], kind="stable").reset_index(drop=True)

        return df

    def create_calendar_features(self, df):
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_month"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype("int16")
        df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")

        # if column exist, replace NaN with "no_event".
        for column in ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]:
            if column in df.columns:
                df[column] = df[column].fillna("no_event").astype("category")

        return df

    def create_lag_features(self, df):
        grouped_sales = df.groupby("item_id", sort=False)["sales"]

        for lag in self.lags:
            df[f"sales_lag_{lag}"] = grouped_sales.shift(lag)

        # Shift first so the current target is never in its own features.
        for window in self.rolling_windows:
            feature_name = f"sales_rolling_mean_{window}"

            df[feature_name] = grouped_sales.transform(
                lambda series: (series.shift(1)
                                .rolling(window=window, min_periods=window,)
                                .mean()
                                ))

        return df

    def create_price_features(self, df):
        df["sell_price"] = pd.to_numeric(df["sell_price"], errors="raise")

        if (df["sell_price"].dropna() < 0).any():
            raise ValueError("Selling price cannot be negative.")

        # Price change versus the previous observation in this series.
        previous_price = df.groupby("item_id", sort=False)["sell_price"].shift(1)

        df["price_change_pct"] = (df["sell_price"].div(previous_price).sub(1)
                                  .replace([np.inf, -np.inf], np.nan))

        return df

    def create_features(self, df):

        df = self.prepare_data(df)

        df = self.create_calendar_features(df)

        df = self.create_lag_features(df)

        df = self.create_price_features(df)

        # Drop warm-up rows without sufficient sales history.
        lag_columns = [f"sales_lag_{lag}" for lag in self.lags]

        rolling_columns = [f"sales_rolling_mean_{window}" for window in self.rolling_windows]

        history_columns = lag_columns + rolling_columns

        before = len(df)

        df = df.dropna(subset=history_columns).reset_index(drop=True)

        logger.info("Removed %s warm-up rows; remaining rows: %s", before - len(df), len(df))

        return df

    def save_features(self, df):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(self.output_path, index=False, engine="pyarrow", compression="snappy")

        logger.info("Saved features to %s with shape %s", self.output_path, df.shape)

        return self.output_path

    def initiate_feature_engineering(self):
        df = self.load_data()

        features = self.create_features(df)

        output_path = self.save_features(features)

        return output_path