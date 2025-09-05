import asyncio
import re
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from supabase import create_client, Client

# -------------------- Supabase Setup --------------------
SUPABASE_URL = "https://uiiuiidynbqtgmqvwvox.supabase.co"   # replace
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVpaXVpaWR5bmJxdGdtcXZ3dm94Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1Njk3ODY2NCwiZXhwIjoyMDcyNTU0NjY0fQ.9BL3prhYuDLi8M_mZ1gce5qUwYc8D7LknwnEppajH4A"             # replace
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

async def get_suspicious_keywords():
    """Fetch suspicious keywords dynamically from Supabase"""
    data = supabase.table("suspicious_keywords").select("keyword").execute()
    if data.data:
        return [row["keyword"].lower() for row in data.data]
    return []

# -------------------- Fraud Detection Logic --------------------
async def fraud_score(message: str):
    msg = message.lower()
    words = re.findall(r"\b\w+\b", msg)

    keywords = await get_suspicious_keywords()  # fetch latest list
    score = 0
    matched = []

    for keyword in keywords:
        if keyword in msg:
            score += 1
            matched.append(keyword)
        else:
            for word in words:
                if keyword == word:
                    score += 1
                    matched.append(keyword)

    return score, list(set(matched))

async def analyze_message(message: str):
    score, matched = await fraud_score(message)
    if score == 0:
        return "✅ Safe: No suspicious content found"
    elif score <= 2:
        return f"⚠️ Caution: Fraud Score {score}. Suspicious terms: {matched}"
    else:
        return f"🚨 Alert: Fraud Score {score}. Highly suspicious!\nMatched terms: {matched}"

# -------------------- Bot Handlers --------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hi! I am the *Fraud Detector Bot*.\n\n"
        "My suspicious keywords are loaded dynamically from Supabase.\n"
        "Send me a message, and I’ll check it."
    )

async def auto_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    if user_text:
        result = await analyze_message(user_text)
        await update.message.reply_text(result)

# -------------------- Async Main --------------------
async def main():
    TOKEN = "8403500260:AAEe8T6MMZyqgWe4la-rfMuEP7eBy01wDNw"  # replace with your bot token
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, auto_check))

    print("🚀 Fraud Detector Bot is running...")

    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        print("🛑 Stopping bot...")
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
