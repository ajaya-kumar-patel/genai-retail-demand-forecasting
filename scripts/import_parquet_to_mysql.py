import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "sales_merged.parquet"

load_dotenv(PROJECT_ROOT / ".env")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    required_env = [
        "MYSQL_HOST",
        "MYSQL_DATABASE",
        "MYSQL_USER",
        "MYSQL_PASSWORD",
    ]
    missing = [key for key in required_env if not os.getenv(key)]
    if missing:
        raise ValueError(f"Missing environment variables: {missing}")

    url = URL.create(
        drivername="mysql+pymysql",
        username=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
        host=os.environ["MYSQL_HOST"],
        port=int(os.getenv("MYSQL_PORT", "3306")),
        database=os.environ["MYSQL_DATABASE"],
    )

    engine = create_engine(url, pool_pre_ping=True)

    try:
        df = pd.read_parquet(DATA_PATH)

        print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
        print("Columns:", list(df.columns))

        # Import in chunks to avoid holding the entire table in one INSERT.
        df.to_sql(
            name="sales",
            con=engine,
            if_exists="fail",
            index=False,
            chunksize=1_000,
            method="multi",
        )

        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM sales")
            ).scalar_one()

            print(f"Imported {count:,} rows into MySQL table 'sales'.")

            sample = conn.execute(
                text("SELECT * FROM sales LIMIT 5")
            ).fetchall()

            for row in sample:
                print(row)

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
