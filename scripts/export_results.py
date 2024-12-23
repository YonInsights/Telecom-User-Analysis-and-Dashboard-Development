from sqlalchemy import create_engine

def export_to_mysql(data, table_name, database_url):
    """
    Export data to a MySQL database.

    Args:
        data (pd.DataFrame): Data to export.
        table_name (str): Target table name.
        database_url (str): MySQL connection string.
    """
    engine = create_engine(database_url)
    data.to_sql(table_name, con=engine, if_exists='replace', index=False)
