import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:IveyIvey@localhost:5433/hiregenius")

print(f"Connecting to database at {DATABASE_URL}...")
engine = create_engine(DATABASE_URL)

try:
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS match_results CASCADE;"))
        conn.commit()
        print("✅ Table 'match_results' dropped successfully.")
        print("It will be automatically recreated with the correct columns when you start main.py!")
except Exception as e:
    print(f"❌ Error: {e}")
