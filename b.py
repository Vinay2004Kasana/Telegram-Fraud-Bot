from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Expanded Suspicious keywords/phrases
suspicious_keywords = [
    # Guaranteed returns
    "guaranteed", "sure shot", "100% return", "no risk", "risk free",
    "profit assured", "fixed income", "safe returns", "next big thing",

    # Pump & dump
    "pump", "dump", "moon", "moonshot", "pump signal", "signal call",
    "target hit", "big target", "buy now", "sell now", "entry call",
    "exit call", "premium call", "jackpot", "multibagger", "rocket stock",

    # Urgency tactics
    "join fast", "limited time", "act now", "don’t miss", "hurry up",
    "last chance", "only today", "fast profit", "quick money",

    # Unrealistic promises
    "double money", "triple money", "10x returns", "guaranteed profit",
    "risk-free investment", "sure profit", "100 percent confidence",

    # Insider claims
    "insider tip", "secret stock", "hidden gem", "exclusive call",
    "inside info", "hot tip", "sure insider", "confidential tip",

    # FOMO & hype
    "next big stock", "lifetime opportunity", "don’t miss this",
    "get rich quick", "jackpot call", "multibagger alert", "money rain",
    "next jackpot", "super hit stock"
]

def fraud_score(message: str) -> int:
    msg = message.lower()
    score = 0
    matched = []
    for keyword in suspicious_keywords:
        if keyword in msg:   # detect single words & phrases
            score += 1
            matched.append(keyword)
    return score, matched

def analyze_message(message: str):
    score, matched = fraud_score(message)
    if score == 0:
        return "✅ Safe: No suspicious content found"
    elif score <= 2:
        return f"⚠️ Caution: Fraud Score {score}. Contains suspicious terms: {matched}"
    else:
        return f"🚨 Alert: Fraud Score {score}. Highly suspicious message!\nMatched terms: {matched}"

# -------------------- Bot Handlers --------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hi! I am the *Fraud Detector Bot*.\n\n"
        "Forward me any Telegram/WhatsApp stock tip message and I’ll check if it looks fraudulent.",
        parse_mode="Markdown"
    )

async def check_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    if user_text:
        result = analyze_message(user_text)
        await update.message.reply_text(result)

# -------------------- Main --------------------
def main():
    TOKEN = "8403500260:AAEe8T6MMZyqgWe4la-rfMuEP7eBy01wDNw"  # ⚠️ Replace with your own BotFather token
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, check_message))

    print("🚀 Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
