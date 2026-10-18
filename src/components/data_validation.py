
import json
import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


class DataValidation:
    """Validate the merged retail sales dataset."""

    REQUIRED_COLUMNS = {
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "d",
        "date",
        "wm_yr_wk",
        "sales",
        "sell_price",
        "snap_CA",
    }

    SNAP_COLUMNS = [
        column
        for column in ["snap_CA", "snap_TX", "snap_WI"]
    ]

    def __init__(self, data_path=None):
        self.project_root = Path(__file__).resolve().parents[2]

        self.data_path = (
            Path(data_path)
            if data_path is not None
            else self.project_root
            / "data"
            / "processed"
            / "sales_merged.parquet"
        )

        self.report_path = (
            self.project_root
            / "data"
            / "processed"
            / "validation_report.json"
        )

    def validate(self):
        """Run validation checks and save a JSON report."""

        if not self.data_path.is_file():
            raise FileNotFoundError(
                f"Processed dataset not found: {self.data_path}"
            )

        logger.info("Loading dataset from %s", self.data_path)

        df = pd.read_parquet(self.data_path)

        if df.empty:
            raise ValueError("Processed dataset is empty.")

        errors = []
        warnings = []

        # 1. Schema validation
        missing_columns = self.REQUIRED_COLUMNS - set(df.columns)

        if missing_columns:
            errors.append(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        if errors:
            self._save_report(
                df=df,
                errors=errors,
                warnings=warnings,
            )

            raise ValueError("; ".join(errors))

        # 2. Critical null checks
        critical_columns = [
            "id",
            "item_id",
            "store_id",
            "d",
            "date",
            "wm_yr_wk",
            "sales",
        ]

        critical_nulls = (
            df[critical_columns]
            .isna()
            .sum()
            .to_dict()
        )

        for column, count in critical_nulls.items():
            if count:
                errors.append(
                    f"{column} contains {count} missing values."
                )

        # 3. Duplicate item-date checks
        duplicate_count = int(
            df.duplicated(subset=["id", "d"]).sum()
        )

        if duplicate_count:
            errors.append(
                f"Found {duplicate_count} duplicate item-date records."
            )

        # 4. Sales validation
        numeric_sales = pd.to_numeric(
            df["sales"],
            errors="coerce",
        )

        if numeric_sales.isna().any():
            errors.append("Sales contains non-numeric values.")
        else:
            if (numeric_sales < 0).any():
                errors.append("Sales contains negative values.")

            if not (numeric_sales % 1 == 0).all():
                errors.append(
                    "Sales contains non-integer values."
                )

        # 5. Date validation
        parsed_dates = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

        if parsed_dates.isna().any():
            errors.append(
                "Some date values cannot be parsed."
            )

        # Check whether the calendar key and date agree.
        day_numbers = (
            df["d"]
            .astype("string")
            .str.extract(r"^d_(\d+)$", expand=False)
        )

        if day_numbers.isna().any():
            errors.append(
                "Some d values do not match the expected d_N format."
            )

        # 6. Price validation
        prices = pd.to_numeric(
            df["sell_price"],
            errors="coerce",
        )

        invalid_prices = prices.notna() & (prices < 0)

        if invalid_prices.any():
            errors.append(
                "Selling prices contain negative values."
            )

        missing_price_count = int(prices.isna().sum())

        if missing_price_count:
            warnings.append(
                f"{missing_price_count} rows have missing selling prices. "
                "Investigate these before feature engineering; "
                "do not automatically replace them with zero."
            )

        # 7. SNAP indicator validation
        snap_columns = [
            column
            for column in self.SNAP_COLUMNS
            if column in df.columns
        ]

        for column in snap_columns:
            invalid_snap = ~df[column].isin([0, 1])

            if invalid_snap.any():
                errors.append(
                    f"{column} contains values other than 0 or 1."
                )

        # 8. Dataset coverage
        unique_series = int(df["id"].nunique())
        unique_dates = int(df["date"].nunique())

        if unique_series < 2:
            warnings.append(
                "Fewer than two item-store series found. "
                "Verify that the intended dataset was ingested."
            )

        if unique_dates < 30:
            warnings.append(
                "Fewer than 30 unique dates found. "
                "Verify the dataset's time coverage."
            )

        # 9. Validate item/store identifiers
        if df.groupby("id")["item_id"].nunique().gt(1).any():
            errors.append(
                "An id maps to multiple item_id values."
            )

        if df.groupby("id")["store_id"].nunique().gt(1).any():
            errors.append(
                "An id maps to multiple store_id values."
            )

        self._save_report(
            df=df,
            errors=errors,
            warnings=warnings,
        )

        if errors:
            raise ValueError(
                f"Data validation failed: {'; '.join(errors)}. "
                f"Check {self.report_path}"
            )

        logger.info(
            "Data validation passed with %s warning(s).",
            len(warnings),
        )

        return {
            "status": "passed",
            "rows": len(df),
            "unique_series": unique_series,
            "unique_dates": unique_dates,
            "warnings": warnings,
            "data_path": str(self.data_path),
        }

    def _save_report(self, df, errors, warnings):
        """Save validation results for reproducibility."""

        self.report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report = {
            "dataset": str(self.data_path),
            "rows": len(df),
            "columns": list(df.columns),
            "errors": errors,
            "warnings": warnings,
            "status": "failed" if errors else "passed",
        }

        with open(self.report_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        logger.info(
            "Validation report saved to %s",
            self.report_path,
        )
