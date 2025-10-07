# create_tables.py

from .sql_db import db, app
from sqlalchemy import text, inspect
from .model import Game, Sample



if __name__ == '__main__':
    with app.app_context():
        
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        print("Tables found:", tables)
        try:
            result = db.session.execute(text("SELECT current_database();"))
            current_db = list(result)[0][0]
            print(f"Current DB: {current_db}")
        except Exception as e:
            print(f"DB Check failed: {e}")

        db.create_all() #only creates tables for imported models and are bound to the same db instance
        print("Tables created.")
