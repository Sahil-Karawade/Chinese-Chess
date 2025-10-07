#" Connecting Flask to Pgsql"

# db.py
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from dotenv import load_dotenv


#Load env
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Update this with your actual password
POSTGRES_USER = os.getenv("DB_USER")
POSTGRES_PW = os.getenv("DB_PASS") 
POSTGRES_URL = os.getenv("DB_URL")
POSTGRES_DB = os.getenv("DB_NAME")
                    

# PostgreSQL connection URI
DB_URI = f'postgresql://{POSTGRES_USER}:{POSTGRES_PW}@{POSTGRES_URL}/{POSTGRES_DB}'

# Configure SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Create the SQLAlchemy db object
db = SQLAlchemy(app)

# test_db_connection.py (optional)


with app.app_context():
    try:
        with db.engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
        
        print("Connected to PostgreSQL successfully.")
    except Exception as e:
        print("Connection failed:", e)
