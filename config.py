# config.py

from supabase import create_client, Client

# -------------------- Telegram Bot Token --------------------
TELEGRAM_BOT_TOKEN = "8403500260:AAEe8T6MMZyqgWe4la-rfMuEP7eBy01wDNw"   # ⚠️ replace with your BotFather token

# -------------------- Supabase Setup --------------------
SUPABASE_URL = "https://uiiuiidynbqtgmqvwvox.supabase.co"   # replace
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVpaXVpaWR5bmJxdGdtcXZ3dm94Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1Njk3ODY2NCwiZXhwIjoyMDcyNTU0NjY0fQ.9BL3prhYuDLi8M_mZ1gce5qUwYc8D7LknwnEppajH4A"             # replace

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_supabase():
    """Return supabase client"""
    return supabase
