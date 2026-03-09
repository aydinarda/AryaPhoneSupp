# supabase_db.py
import os
import streamlit as st
from supabase import create_client

@st.cache_resource
def get_client():
    url = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL / SUPABASE_ANON_KEY in Streamlit secrets.")
    return create_client(url, key)

def insert_submission(payload: dict):
    client = get_client()
    return client.table("submissions").insert(payload).execute()

def fetch_all_submissions(limit: int = 5000):
    client = get_client()
    return client.table("submissions").select("*").order("created_at", desc=True).limit(limit).execute()