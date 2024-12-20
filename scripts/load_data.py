import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
def fetch_data(query: str) -> pd.DataFrame:
    """
    Connect to the database and execute a query.

    Args:
        query (str): SQL query to execute.

    Returns:
        pd.DataFrame: DataFrame containing query results.
    """
    try:
        # Read database credentials from environment variables
        db_config = {
            "dbname": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT")
        }

        # Establish connection
        with psycopg2.connect(**db_config) as conn:
            # Execute query and fetch results into a DataFrame
            df = pd.read_sql_query(query, conn)
            return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()
