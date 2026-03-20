# Arya Fullstack App

This folder contains a server/client version of the app migrated from the existing Streamlit implementation without changing the core game logic.

## Structure

- `server/app/main.py`: FastAPI backend
- `server/app/service.py`: Service layer that uses MinCostAgent logic
- `server/app/db.py`: Supabase connection (same `submissions` table)
- `client/`: Simple web client (HTML/CSS/JS)

## Run Locally

1. Go to the `arya_fullstack_app/server` folder.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Start the backend:
   - `uvicorn app.main:app --reload --port 8000`
4. Open in browser:
   - `http://127.0.0.1:8000`

## Notes

- The backend imports the optimization controller module.
- For DB credentials, environment variables (`SUPABASE_URL`, `SUPABASE_ANON_KEY`) are checked first.
- If environment variables are missing, values are read from the root `secrets.toml` file.
