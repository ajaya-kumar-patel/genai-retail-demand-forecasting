
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

load_dotenv(PROJECT_ROOT / ".env")

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "retail_forecasting")

BATCH_SIZE = 500

# CSV filename -> MongoDB collection name
DATASETS = {
    "sales_history.csv": "sales_history",
    "calendar.csv": "calendar",
    "sell_prices.csv": "sell_prices",
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def get_unique_key(collection_name: str) -> list[str]:
    """Return the natural key for each dataset."""

    keys = {
        "sales_history": ["item_id", "store_id"],
        "calendar": ["d"],
        "sell_prices": ["store_id", "item_id", "wm_yr_wk"],
    }

    return keys[collection_name]


def validate_file(file_path: Path, collection_name: str) -> None:
    """Check file existence and required columns."""

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}\n"
                                "Run your data preparation script first.")

    required_columns = {
        "sales_history": {"item_id", "store_id"},
        "calendar": {"d"},
        "sell_prices": {"store_id", "item_id", "wm_yr_wk"},
    }

    # Read only the header for an initial schema check.
    columns = set(pd.read_csv(file_path, nrows=0).columns)
    missing = required_columns[collection_name] - columns

    if missing:
        raise ValueError(f"{file_path.name} is missing required columns: {missing}")


def create_indexes(db) -> None:
    """Create indexes to identify records consistently."""

    for collection_name in DATASETS.values():
        key_fields = get_unique_key(collection_name)
        index_spec = [(field, ASCENDING) for field in key_fields]

        db[collection_name].create_index(index_spec, unique=True, name=f"unique_{collection_name}_key")


def upload_csv(file_path: Path, collection, collection_name: str) -> None:
    """
    Upload CSV rows in batches.

    Existing documents with the same natural key are replaced,
    so rerunning the script updates records instead of duplicating them.
    Documents not present in the CSV are not automatically deleted.
    """

    from pymongo import ReplaceOne

    unique_key = get_unique_key(collection_name)
    total_processed = 0

    for chunk in pd.read_csv(file_path, chunksize=BATCH_SIZE):
        # Replace pandas NaN values with MongoDB-compatible nulls.
        chunk = chunk.astype(object).where(pd.notna(chunk), None)

        operations = []

        for record in chunk.to_dict(orient="records"):
            # MongoDB supports BSON types, so convert NumPy scalars
            # to standard Python values where needed.
            record = {
                key: (
                    value.item()
                    if hasattr(value, "item")
                    else value
                )
                for key, value in record.items()
            }

            # All natural-key fields must be present.
            if any(record.get(key) is None for key in unique_key):
                raise ValueError(
                    f"Missing natural key in {collection_name}: "
                    f"{unique_key}"
                )

            filter_query = {key: record[key] for key in unique_key}

            operations.append(ReplaceOne(filter_query, record, upsert=True))

        if operations:
            result = collection.bulk_write(operations, ordered=False,)
            total_processed += len(operations)

        print(
            f"\r{collection_name}: "
            f"{total_processed:,} rows processed",
            end="",
            flush=True,
        )

    print()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    if not MONGODB_URI:
        raise ValueError("MONGODB_URI is missing. Add it to your project .env file.")

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Processed data directory not found: {DATA_DIR}")

    print("Validating datasets...")

    for filename, collection_name in DATASETS.items():
        validate_file(DATA_DIR / filename, collection_name)

    client = None

    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=15000)

        # Confirm the connection before starting the upload.
        client.admin.command("ping")
        print("Connected to MongoDB Atlas.")

        db = client[MONGODB_DATABASE]

        # Create indexes before writing records.
        create_indexes(db)

        for filename, collection_name in DATASETS.items():
            file_path = DATA_DIR / filename
            collection = db[collection_name]

            print(f"\nUploading {filename} -> {collection_name}")

            upload_csv(file_path=file_path, collection=collection, collection_name=collection_name)

            print(f"Documents currently in {collection_name}: "
                  f"{collection.count_documents({}):,}")

        print("\nUpload completed successfully.")

    except (PyMongoError, OSError, ValueError) as exc:
        print(f"\nUpload failed: {exc}", file=sys.stderr)
        raise

    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()