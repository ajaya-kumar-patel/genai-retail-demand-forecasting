
import logging
from pathlib import Path

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.feature_engineering import FeatureEngineering
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluator

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Run selected stages of the forecasting workflow."""

    def __init__(
        self,
        run_ingestion=False,
        run_validation=False,
        run_feature_engineering=False,
        run_training=False,
        run_evaluation=False,
        validation_days=28,
    ):
        self.project_root = Path(__file__).resolve().parents[2]
        self.processed_dir = self.project_root / "data" / "processed"

        self.sales_path = self.processed_dir / "sales_merged.parquet"
        self.features_path = self.processed_dir / "features.parquet"
        self.model_path = self.project_root / "models" / "xgboost_pipeline.joblib"

        self.run_ingestion = run_ingestion
        self.run_validation = run_validation
        self.run_feature_engineering = run_feature_engineering
        self.run_training = run_training
        self.run_evaluation = run_evaluation
        self.validation_days = validation_days

        if validation_days < 1:
            raise ValueError("validation_days must be positive.")

        if run_evaluation and not run_training and not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Train a model first or provide an existing model."
            )

    def run(self):
        results = {}

        try:
            # STEP 1: DATA INGESTION
            if self.run_ingestion:
                logger.info("STEP 1: DATA INGESTION")

                ingestion = DataIngestion()
                output_path = ingestion.initiate_data_ingestion()

                self.sales_path = Path(output_path)
                results["ingestion_path"] = str(self.sales_path)

            # STEP 2: DATA VALIDATION
            if self.run_validation:
                logger.info("STEP 2: DATA VALIDATION")

                validator = DataValidation()
                validation_result = validator.validate()

                results["validation"] = validation_result

                if validation_result.get("status") != "passed":
                    raise ValueError(
                        "Data validation failed. "
                        "The remaining pipeline stages were stopped."
                    )

            # STEP 3: FEATURE ENGINEERING
            if self.run_feature_engineering:
                logger.info("STEP 3: FEATURE ENGINEERING")

                if not self.sales_path.is_file():
                    raise FileNotFoundError(
                        f"Input dataset not found: {self.sales_path}. "
                        "Run ingestion first or provide the existing dataset."
                    )

                feature_engineer = FeatureEngineering(
                    input_path=self.sales_path
                )
                output_path = (
                    feature_engineer.initiate_feature_engineering()
                )

                self.features_path = Path(output_path)
                results["features_path"] = str(self.features_path)

            # STEP 4: MODEL TRAINING
            if self.run_training:
                logger.info("STEP 4: MODEL TRAINING")

                if not self.features_path.is_file():
                    raise FileNotFoundError(
                        f"Feature dataset not found: {self.features_path}. "
                        "Run feature engineering first or provide an "
                        "existing features.parquet file."
                    )

                trainer = ModelTrainer(
                    input_path=self.features_path,
                    validation_days=self.validation_days,
                )

                training_result = trainer.initiate_model_training()
                results["training"] = training_result

                self.model_path = Path(training_result["model_path"])

            # STEP 5: MODEL EVALUATION
            if self.run_evaluation:
                logger.info("STEP 5: MODEL EVALUATION")

                if not self.features_path.is_file():
                    raise FileNotFoundError(
                        f"Feature dataset not found: {self.features_path}"
                    )

                if not self.model_path.is_file():
                    raise FileNotFoundError(
                        f"Model not found: {self.model_path}"
                    )

                evaluator = ModelEvaluator(
                    input_path=self.features_path,
                    model_path=self.model_path,
                    validation_days=self.validation_days,
                )

                evaluation_result = (
                    evaluator.initiate_model_evaluation()
                )
                results["evaluation"] = evaluation_result

                metrics = evaluation_result["metrics"]

                logger.info("EVALUATION SUMMARY")
                for name, value in metrics.items():
                    logger.info("%s: %s", name, value)

                logger.info(
                    "Metrics saved to: %s",
                    evaluation_result["metrics_path"],
                )
                logger.info(
                    "Predictions saved to: %s",
                    evaluation_result["predictions_path"],
                )

            if not results:
                logger.info("No stages selected. Nothing was executed.")
            else:
                logger.info("Selected pipeline stages completed successfully.")

            return results

        except Exception:
            logger.exception("Pipeline execution failed.")
            raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    pipeline = TrainingPipeline(
        run_ingestion=False,
        run_validation=False,
        run_feature_engineering=False,
        run_training=False,
        run_evaluation=True,
    )

    pipeline.run()
