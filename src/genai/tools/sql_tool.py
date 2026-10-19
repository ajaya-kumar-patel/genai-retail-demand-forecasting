
import os
from pathlib import Path

import sqlglot
from sqlglot import exp
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from langchain_ollama import ChatOllama

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

MAX_ROWS = 100



llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
)

SCHEMA_DESCRIPTION = """
You have access to one MySQL table named sales.

Columns:
- id: source record identifier (don't use this)
- item_id: product identifier
- dept_id: department identifier
- cat_id: product category
- store_id: store identifier
- state_id: state identifier
- d: sales-day identifier, such as d_1
- sales: units sold
- date: calendar date
- wm_yr_wk: year-week identifier
- event_name_1: primary event name
- event_type_1: primary event type
- event_name_2: secondary event name
- event_type_2: secondary event type
- snap_CA: SNAP indicator
- sell_price: selling price

Rules:
- Generate MySQL 8-compatible SQL.
- Use only the sales table and the columns listed above.
- Generate exactly one read-only SELECT statement.
- Never generate data-modifying statements, including INSERT,
  UPDATE, DELETE, DROP, ALTER, TRUNCATE, or similar operations.
- Do not generate multiple statements or assume nonexistent columns.
- This assistant supports sales analytics only.
- If a request involves modifying or deleting data, do not generate SQL.
- Never suggest alternative ways to modify or delete database data.
- If a question cannot be answered using the available schema,
  explain the limitation instead of inventing data or columns.
- Return SQL only for answerable requests, without Markdown fences
  or explanations.
- For unanswerable requests, return a concise explanation of the
  limitation without generating SQL.
- Profit cannot be calculated because cost data is unavailable.


Business rules:
- A product means one unique item_id.
- When asked for products ranked by sales, group by item_id,
  never cat_id or dept_id.
- "Top products by total units sold" means:
  SELECT item_id, SUM(sales) AS total_sales
  FROM sales
  GROUP BY item_id
  ORDER BY total_sales DESC
  LIMIT 5
- cat_id represents a category, not an individual product.
- dept_id represents a department, not an individual product.
- Only group by cat_id when the user explicitly asks about categories.
- Only group by dept_id when the user explicitly asks about departments.


Date and aggregation rules:
- The date column contains the calendar date for each sales record.
- Use date for all calendar-based questions.
- For daily analysis, use DATE(date).
- For monthly analysis, use DATE_FORMAT(date, '%Y-%m').
- For yearly analysis, use YEAR(date).
- For quarterly analysis, use YEAR(date) and QUARTER(date).
- For day-of-week analysis, use DAYNAME(date).
- When comparing months across multiple years, include the year.
- Use SUM(sales) for total units sold.
- Use SUM(sales * sell_price) for estimated sales revenue.
- Use AVG(sell_price) for average selling price.
- For monthly, daily, or yearly aggregations, group by the appropriate
  date expression.
- For questions asking for the highest or lowest sales period,
  aggregate the data first, then sort by the aggregate and use LIMIT 1.
- "Which month had the highest total sales?" means monthly total
  units sold, grouped by year and month, sorted descending, LIMIT 1.
- "Which month had the highest sales revenue?" means monthly
  SUM(sales * sell_price), grouped by year and month, sorted descending,
  LIMIT 1.
- Never confuse total units sold with sales revenue.

Aggregation and Ranking Rules:
- Total units sold: SUM(sales).
- Estimated sales revenue: SUM(sales * sell_price).
- Average selling price: AVG(sell_price).
- Do not confuse units sold with sales revenue.
- For highest- or lowest-sales-period questions, aggregate by the
  requested period before sorting and applying LIMIT 1.
- "Which month had the highest total sales?" means monthly total
  units sold, grouped by year and month, ordered by units sold
  descending, LIMIT 1.
- "Which month had the highest sales revenue?" means monthly
  estimated revenue, grouped by year and month, ordered by revenue
  descending, LIMIT 1.

Default Product Ranking:
For "Top products by total units sold", use:
SELECT item_id, SUM(sales) AS total_sales
FROM sales
GROUP BY item_id
ORDER BY total_sales DESC
LIMIT 5;

"""



def create_mysql_engine():
    """Create a connection pool using the read-only MySQL account."""
    required = [
        "MYSQL_HOST",
        "MYSQL_DATABASE",
        "MYSQL_READ_USER",
        "MYSQL_READ_PASSWORD",
    ]
    missing = [key for key in required if not os.getenv(key)]

    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")

    url = URL.create(
        drivername="mysql+pymysql",
        username=os.environ["MYSQL_READ_USER"],
        password=os.environ["MYSQL_READ_PASSWORD"],
        host=os.environ["MYSQL_HOST"],
        port=int(os.getenv("MYSQL_PORT", "3306")),
        database=os.environ["MYSQL_DATABASE"],
    )

    return create_engine(url, pool_pre_ping=True, pool_recycle=1800)


def validate_sql(sql: str) -> str:
    """Allow one read-only SELECT query referencing only the sales table."""
    if not sql or not sql.strip():
        raise ValueError("SQL query cannot be empty.")

    statements = sqlglot.parse(sql, read="mysql")

    if len(statements) != 1 or statements[0] is None:
        raise ValueError("Exactly one SQL statement is required.")

    query = statements[0]

    # Reject anything except SELECT, including data-modifying CTEs.
    if not isinstance(query, exp.Select):
        raise ValueError("Only SELECT queries are allowed.")

    # Reject SELECT INTO and locking reads.
    if query.args.get("into") or query.args.get("locks"):
        raise ValueError("SELECT INTO and locking reads are not allowed.")

    # Allow only the sales table, including references inside subqueries.
    for table in query.find_all(exp.Table):
        if (
            table.name.lower() != "sales"
            or table.db
            or table.catalog
        ):
            raise ValueError("Queries may reference only the sales table.")

    # Reject table-valued functions and unsupported FROM sources.
    for node in query.find_all(exp.Func):
        if isinstance(node, exp.Anonymous):
            raise ValueError(
                f"Function '{node.name}' is not allowed."
            )

    # Add a limit if the query doesn't already have one.
    if query.args.get("limit") is None:
        query = query.limit(MAX_ROWS)

    return query.sql(dialect="mysql")


def execute_read_query(engine, sql: str) -> dict:
    """Validate and execute a read-only query."""
    safe_sql = validate_sql(sql)

    with engine.connect() as connection:
        result = connection.execute(text(safe_sql))
        columns = list(result.keys())
        rows = result.fetchmany(MAX_ROWS)

    return {
        "columns": columns,
        "rows": [list(row) for row in rows],
    }



def generate_sql(question: str) -> str:
    """Convert a natural-language question into a validated SQL query."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    prompt = f"""
    You are a MySQL analytics assistant.

    DATABASE SCHEMA AND BUSINESS RULES:
    {SCHEMA_DESCRIPTION}
    
    USER QUESTION:
    {question}
    

    RULES:
    - Product IDs must always be filtered using item_id.
    - Never use cat_id or dept_id to filter a product ID.
    - Average selling price means AVG(sell_price).
    - For a question about one product, use WHERE item_id = 'PRODUCT_ID'.
    - Use only the sales table and its listed columns.
    - Generate one read-only SELECT query.
    - Return SQL only, without Markdown fences.

    Example:
    Question: What is the average selling price of FOODS_3_090?

    SELECT item_id, AVG(sell_price) AS average_selling_price
    FROM sales
    WHERE item_id = 'FOODS_3_090'
    GROUP BY item_id;
    """

    response = llm.invoke(prompt)
    generated_sql = response.content.strip()

    # Remove a Markdown SQL code fence if the model includes one.
    if generated_sql.startswith("```"):
        lines = generated_sql.splitlines()
        if lines and lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        generated_sql = "\n".join(lines).strip()

    # Validate before returning the generated SQL.
    return validate_sql(generated_sql)



def run_sql_tool(question: str) -> dict:
    """Generate, validate, and execute SQL for a natural-language question."""
    sql = generate_sql(question)

    engine = create_mysql_engine()
    try:
        result = execute_read_query(engine, sql)
        return {
            "question": question,
            "sql": sql,
            "columns": result["columns"],
            "rows": result["rows"],
        }
    finally:
        engine.dispose()


if __name__ == "__main__":
    engine = create_mysql_engine()

    try:
        query = """
            SELECT item_id, SUM(sales) AS total_sales
            FROM sales
            GROUP BY item_id
            ORDER BY total_sales DESC
            LIMIT 5
        """

        output = execute_read_query(engine, query)

        print("Columns:", output["columns"])
        for row in output["rows"]:
            print(row)
    finally:
        engine.dispose()
