
import os
import re
from datetime import timedelta

import requests
from sqlalchemy import text

from src.genai.tools.sql_tool import create_mysql_engine


FORECAST_API_URL = os.getenv(
    "FORECAST_API_URL",
    "https://genai-retail-demand-forecasting.onrender.com/predict",
)

MAX_HORIZON = 28


def run_forecast_tool(question: str) -> dict:
    """Forecast one product recursively using the deployed prediction API."""

    # Extract a product ID such as FOODS_3_090.
    item_match = re.search(
        r"\b[A-Z]+_[A-Z0-9]+_[A-Z0-9]+\b",
        question.upper(),
    )

    if not item_match:
        return {
            "error": (
                "Please specify a product ID, for example "
                "'Forecast FOODS_3_090 for the next 7 days.'"
            )
        }

    item_id = item_match.group(0)

    # Default to seven days unless the user specifies another horizon.
    horizon_match = re.search(
        r"\b(\d{1,2})\s+days?\b",
        question.lower(),
    )
    horizon = int(horizon_match.group(1)) if horizon_match else 7

    if not 1 <= horizon <= MAX_HORIZON:
        return {
            "error": f"Forecast horizon must be between 1 and {MAX_HORIZON} days."
        }

    engine = create_mysql_engine()

    try:
        with engine.connect() as connection:
            result = connection.execute(
                text("""
                    SELECT
                        item_id,
                        dept_id,
                        cat_id,
                        DATE(date) AS date,
                        sales,
                        sell_price
                    FROM sales
                    WHERE item_id = :item_id
                    ORDER BY date DESC
                    LIMIT 60
                """),
                {"item_id": item_id},
            )

            rows = [dict(row) for row in result.mappings().all()]

    finally:
        engine.dispose()

    if not rows:
        return {"error": f"No historical data found for {item_id}."}

    # Convert newest-first database results into chronological order.
    history = list(reversed(rows))
    latest = rows[0]
    latest_date = latest["date"]

    if len(history) < 28:
        return {
            "error": (
                f"Insufficient historical data for {item_id}. "
                "At least 28 daily sales records are required."
            )
        }

    # Verify the last 28 observations represent consecutive calendar days.
    recent_dates = [row["date"] for row in history[-28:]]

    for previous, current in zip(recent_dates, recent_dates[1:]):
        if current - previous != timedelta(days=1):
            return {
                "error": (
                    f"Historical dates for {item_id} are not consecutive. "
                    "Cannot safely calculate daily lag features."
                )
            }

    sales_history = [float(row["sales"]) for row in history]

    # Carry forward the latest available non-null selling price.
    latest_price = next(
        (
            float(row["sell_price"])
            for row in rows
            if row["sell_price"] is not None
        ),
        None,
    )

    if latest_price is None:
        return {"error": f"No historical selling price found for {item_id}."}

    forecasts = []

    # Each prediction becomes historical input for the following day.
    for step in range(1, horizon + 1):
        forecast_date = latest_date + timedelta(days=step)

        payload = {
            "item_id": item_id,
            "dept_id": latest["dept_id"],
            "cat_id": latest["cat_id"],
            "day_of_week": forecast_date.weekday(),
            "day_of_month": forecast_date.day,
            "month": forecast_date.month,
            "year": forecast_date.year,
            "week_of_year": forecast_date.isocalendar().week,
            "is_weekend": int(forecast_date.weekday() >= 5),
            "sales_lag_7": sales_history[-7],
            "sales_lag_14": sales_history[-14],
            "sales_lag_28": sales_history[-28],
            "sales_rolling_mean_7": sum(sales_history[-7:]) / 7,
            "sales_rolling_mean_28": sum(sales_history[-28:]) / 28,
            "snap_CA": 0,
            "sell_price": latest_price,
            "price_change_pct": 0.0,
            "event_name_1": "no_event",
            "event_type_1": "no_event",
            "event_name_2": "no_event",
            "event_type_2": "no_event",
        }

        response = requests.post(
            FORECAST_API_URL,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()

        prediction_result = response.json()
        predicted_sales = float(prediction_result["predicted_sales"])

        forecasts.append({
            "date": forecast_date.isoformat(),
            "predicted_sales": round(predicted_sales, 2),
        })

        # Feed this day's prediction into subsequent lag calculations.
        sales_history.append(predicted_sales)

    return {
        "item_id": item_id,
        "historical_data_through": latest_date.isoformat(),
        "forecast_start_date": forecasts[0]["date"],
        "forecast_end_date": forecasts[-1]["date"],
        "horizon_days": horizon,
        "forecasts": forecasts,
        "assumptions": [
            "Forecast starts after the latest historical date.",
            "Selling price is held at the latest observed price.",
            "Future SNAP indicators are set to 0.",
            "Future event features are set to no_event.",
            "Predicted sales are recursively reused as lag inputs.",
        ],
    }
