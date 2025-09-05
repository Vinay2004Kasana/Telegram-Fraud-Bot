import asyncio
import re
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Expanded Suspicious keywords/phrases
suspicious_keywords = [
    "guaranteed", "sure shot", "100% return", "no risk", "risk free",
    "profit assured", "fixed income", "safe returns", "next big thing",
    "pump", "dump", "moon", "moonshot", "pump signal", "signal call",
    "target hit", "big target", "buy now", "sell now", "entry call",
    "exit call", "premium call", "jackpot", "multibagger", "rocket stock",
    "join fast", "limited time", "act now", "don’t miss", "hurry up",
    "last chance", "only today", "fast profit", "quick money",
    "double money", "triple money", "10x returns", "guaranteed profit",
    "risk-free investment", "sure profit", "100 percent confidence",
    "insider tip", "secret stock", "hidden gem", "exclusive call",
    "inside info", "hot tip", "sure insider", "confidential tip",
    "next big stock", "lifetime opportunity", "don’t miss this",
    "get rich quick", "jackpot call", "multibagger alert", "money rain",
    "next jackpot", "super hit stock"
]

# -------------------- Fraud Detection Logic --------------------
def fraud_score(message: str):
    msg = message.lower()
    words = re.findall(r"\b\w+\b", msg)  # split into words
    score = 0
    matched = []

    for keyword in suspicious_keywords:
        if keyword in msg:  # phrase found in full text
            score += 1
            matched.append(keyword)
        else:
            for word in words:  # check single words
                if keyword == word:
                    score += 1
                    matched.append(keyword)

    return score, list(set(matched))

def analyze_message(message: str):
    score, matched = fraud_score(message)
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
        "Send me any message or add me to a group — I’ll flag suspicious stock/crypto tips.",
        parse_mode="Markdown"
    )

async def auto_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    if user_text:
        result = analyze_message(user_text)
        await update.message.reply_text(result)

# -------------------- Async Main (Clean Shutdown) --------------------
async def main():
    TOKEN = "8403500260:AAEe8T6MMZyqgWe4la-rfMuEP7eBy01wDNw"  # ⚠️ Replace with your BotFather token
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, auto_check))

    print("🚀 Fraud Detector Bot is running...")

    # Start bot properly
    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    try:
        await asyncio.Event().wait()  # keep alive
    except (KeyboardInterrupt, SystemExit):
        print("🛑 Stopping bot...")
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
