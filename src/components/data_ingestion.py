
import logging
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError


logger = logging.getLogger(__name__)


class DataIngestion:
    """Extract and prepare retail sales data from MongoDB."""

    def __init__(self, batch_size=500):
        self.project_root = Path(__file__).resolve().parents[2]

        load_dotenv(self.project_root / ".env")

        self.mongodb_uri = os.getenv("MONGODB_URI")
        self.database_name = os.getenv("MONGODB_DATABASE", "retail_forecasting")

        self.batch_size = batch_size

        self.processed_dir = (self.project_root / "data" / "processed")

        self.output_path = (self.processed_dir / "sales_merged.parquet")

        if not self.mongodb_uri:
            raise ValueError("MONGODB_URI is missing from the project .env file.")

    def connect(self):
        """Connect to MongoDB and return the client and database."""

        client = MongoClient(self.mongodb_uri, serverSelectionTimeoutMS=15000,)

        client.admin.command("ping")

        db = client[self.database_name]

        logger.info("Connected to MongoDB database: %s", self.database_name,)

        return client, db

    def load_calendar(self, db):
        """Load the relatively small calendar collection."""

        records = list(
            db["calendar"].find({}, {"_id": 0})
        )

        if not records:
            raise ValueError("MongoDB calendar collection is empty.")

        calendar = pd.DataFrame(records)

        required = {"d", "wm_yr_wk"}

        if not required.issubset(calendar.columns):
            raise ValueError(f"Calendar is missing columns: " f"{required - set(calendar.columns)}")

        if calendar["d"].isna().any():
            raise ValueError("Calendar contains missing date keys.")

        if calendar["d"].duplicated().any():
            raise ValueError("Calendar contains duplicate date keys.")

        logger.info("Loaded %s calendar records", len(calendar))

        return calendar

    def get_sales_cursor(self, db):
        """Return a cursor over sales documents."""

        return db["sales_history"].find(
            {},
            {"_id": 0},
            batch_size=self.batch_size,
        )

    def transform_sales_to_long(self, sales_batch):
        """Convert wide daily sales columns into long format."""

        sales = pd.DataFrame(sales_batch)

        if sales.empty:
            return pd.DataFrame()

        required = {"item_id", "store_id", "d_1",}

        if not required.issubset(sales.columns):
            raise ValueError(
                "Sales collection is missing required fields. "
                f"Required fields include: {sorted(required)}"
            )

        id_columns = [
            column
            for column in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
            if column in sales.columns
        ]

        day_columns = [
            column
            for column in sales.columns
            if column.startswith("d_")
        ]

        if not day_columns:
            raise ValueError("No daily sales fields beginning with 'd_' found.")

        sales_long = sales.melt(
            id_vars=id_columns,
            value_vars=day_columns,
            var_name="d",
            value_name="sales",
        )

        return sales_long

    def load_prices_for_batch(self, db, sales_batch):
        """Load prices only for item-store pairs in this batch."""

        pairs = sales_batch[["item_id", "store_id"]].drop_duplicates()

        conditions = [
            {
                "item_id": row.item_id,
                "store_id": row.store_id
            }
            for row in pairs.itertuples(index=False)
        ]

        if not conditions:
            return pd.DataFrame(columns=["item_id", "store_id", "wm_yr_wk", "sell_price"])

        projection = {
            "_id": 0,
            "item_id": 1,
            "store_id": 1,
            "wm_yr_wk": 1,
            "sell_price": 1,
        }

        records = db["sell_prices"].find(
            {"$or": conditions},
            projection,
            batch_size=5000,
        )

        prices = pd.DataFrame(list(records))

        if prices.empty:
            return prices

        price_keys = ["store_id", "item_id", "wm_yr_wk"]

        if prices.duplicated(price_keys).any():
            raise ValueError(
                "Duplicate item-store-week keys found in sell_prices."
            )

        return prices

    def merge_batch(self, sales_batch, calendar, db):
        """Merge one sales batch with calendar and price information."""

        sales_long = self.transform_sales_to_long(sales_batch)

        if sales_long.empty:
            return sales_long

        sales_long = sales_long.merge(calendar, on="d", how="left",
                                      validate="many_to_one", indicator="_calendar_match")

        if not (sales_long["_calendar_match"] == "both").all():
            raise ValueError("Some sales dates do not match the calendar.")

        sales_long = sales_long.drop(columns="_calendar_match")

        prices = self.load_prices_for_batch(db, sales_long)

        if prices.empty:
            sales_long["sell_price"] = float("nan")
        else:
            sales_long = sales_long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left",
                                          validate="many_to_one")

        return sales_long

    def initiate_data_ingestion(self):
        """Extract, transform and save the complete dataset."""

        self.processed_dir.mkdir(parents=True, exist_ok=True,)

        client = None
        writer = None
        total_rows = 0
        batch_number = 0

        try:
            client, db = self.connect()
            calendar = self.load_calendar(db)
            cursor = self.get_sales_cursor(db)

            try:
                while True:
                    documents = []

                    for _ in range(self.batch_size):
                        try:
                            documents.append(next(cursor))
                        except StopIteration:
                            break

                    if not documents:
                        break

                    batch_number += 1

                    sales_batch = pd.DataFrame(documents)

                    merged_batch = self.merge_batch(sales_batch, calendar, db)

                    if merged_batch.empty:
                        continue

                    # Preserve a consistent output schema across batches.
                    table = pa.Table.from_pandas(merged_batch, preserve_index=False)

                    if writer is None:
                        writer = pq.ParquetWriter(self.output_path, table.schema, compression="snappy")

                    writer.write_table(table)

                    total_rows += len(merged_batch)

                    logger.info(
                        "Batch %s completed | "
                        "Rows written: %s",
                        batch_number,
                        f"{total_rows:,}",
                    )

            finally:
                cursor.close()

            if writer is None:
                raise ValueError(
                    "No sales data was processed. "
                    "Check the sales_history collection."
                )

            logger.info(
                "Data ingestion completed. "
                "Total rows: %s | Output: %s",
                f"{total_rows:,}",
                self.output_path,
            )

            return self.output_path

        except (PyMongoError, OSError, ValueError):
            logger.exception("Data ingestion failed")
            raise

        finally:
            if writer is not None:
                writer.close()

            if client is not None:
                client.close()