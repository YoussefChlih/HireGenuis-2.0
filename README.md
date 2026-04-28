# HireGenius Backend

This repository contains the backend and database layer for HireGenius only. It intentionally excludes the frontend and other unrelated project assets.

## Contents

- `main.py` - FastAPI application for CV and job description matching
- `requirements.txt` - Python dependencies
- `schema.sql` - PostgreSQL schema
- `fix_db.py` - Helper script to reset the `match_results` table when needed

## Requirements

- Python 3.8+
- PostgreSQL
- Optional: Google Gemini API key for AI feedback

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Create the database and apply the schema:

```bash
psql -U postgres -c "CREATE DATABASE hiregenius;"
psql -U postgres -d hiregenius -f schema.sql
```

3. Configure environment variables in a local `.env` file:

```env
DATABASE_URL=postgresql://postgres:your_password@localhost:5433/hiregenius
GEMINI_API_KEY=your_gemini_api_key
FRONTEND_ORIGIN=http://localhost:3000
```

4. Start the API:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Notes

- The application creates and migrates tables automatically on startup.
- Do not commit `.env` or other machine-specific files.
- If you only want the backend in GitHub, keep the repository root at this folder and avoid adding frontend files.
