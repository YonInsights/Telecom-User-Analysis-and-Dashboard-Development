import psycopg2
import sys
sys.path.insert(0, 'src')
from db_config import DATABASE_CONFIG

def test_connection():
    try:
        print("Connecting to the database...")
        conn = psycopg2.connect(
            host=DATABASE_CONFIG["host"],
            port=DATABASE_CONFIG["port"],
            user=DATABASE_CONFIG["user"],
            password=DATABASE_CONFIG["password"],
            database=DATABASE_CONFIG["database"]
        )
        print("Connection successful!")
        conn.close()
    except Exception as e:
        print("Connection failed.")
        print(f"Error: {e}")

if __name__ == "__main__":
    test_connection()
