import joblib
import numpy as np
import pytest

from src.pipelines.prediction_pipeline import PredictionPipeline


class DummyModel:
    """Small deterministic model used only for testing."""

    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, input_df):
        assert len(input_df) == 1
        return np.array([self.prediction])


@pytest.fixture
def prediction_args():
    return {
        "item_id": "HOBBIES_1_001",
        "dept_id": "HOBBIES_1",
        "cat_id": "HOBBIES",
        "day_of_week": 3,
        "day_of_month": 15,
        "month": 11,
        "year": 2016,
        "week_of_year": 46,
        "is_weekend": 0,
        "sales_lag_7": 4,
        "sales_lag_14": 6,
        "sales_lag_28": 5,
        "sales_rolling_mean_7": 4.7,
        "sales_rolling_mean_28": 5.2,
        "snap_CA": 0,
        "sell_price": 9.97,
        "price_change_pct": 0.0,
        "event_name_1": "no_event",
        "event_type_1": "no_event",
        "event_name_2": "no_event",
        "event_type_2": "no_event",
    }


@pytest.fixture
def make_pipeline(tmp_path):
    def create(prediction):
        model_path = tmp_path / "dummy_model.joblib"
        joblib.dump(DummyModel(prediction), model_path)

        return PredictionPipeline(model_path=model_path)

    return create


def test_prediction_returns_expected_structure(make_pipeline, prediction_args):
    pipeline = make_pipeline(12.5)

    result = pipeline.predict(**prediction_args)

    assert result == {
        "item_id": "HOBBIES_1_001",
        "predicted_sales": 12.5,
    }


def test_negative_prediction_is_clipped_to_zero(make_pipeline, prediction_args):
    pipeline = make_pipeline(-4.0)

    result = pipeline.predict(**prediction_args)

    assert result["predicted_sales"] == 0.0


def test_missing_model_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model not found"):
        PredictionPipeline(model_path=tmp_path / "missing_model.joblib")


# def test_missing_model_raises_file_not_found(tmp_path, prediction_args):
#     pipeline = PredictionPipeline(
#         model_path=tmp_path / "missing_model.joblib"
#     )

#     with pytest.raises(FileNotFoundError, match="Model not found"):
#         pipeline.predict(**prediction_args)


def test_missing_model_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model not found"):
        PredictionPipeline(model_path=tmp_path / "missing_model.joblib")


def test_prediction_is_a_non_negative_number(make_pipeline, prediction_args):
    pipeline = make_pipeline(7.25)

    result = pipeline.predict(**prediction_args)

    assert isinstance(result["predicted_sales"], float)
    assert result["predicted_sales"] >= 0
