from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from supabase import create_client, Client
import os

# -------------------- Supabase Setup --------------------
SUPABASE_URL = "https://xygdqapjctesdivpgntw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inh5Z2RxYXBqY3Rlc2RpdnBnbnR3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTY0NDY2NTQsImV4cCI6MjA3MjAyMjY1NH0.w95J21AzcKp5G5WAZiz7LQD8pUWbTKeGOzVFe8edYlU"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------- ML Setup --------------------
fraudulent_msgs = [
    "100% guaranteed return buy now!",
    "Sure shot call buy stock at 50 target 70",
    "Double money in 1 week",
    "Risk free insider tip buy immediately",
    "Join fast jackpot stock today"
]

safe_msgs = [
    "Stock market involves risk, do your own research",
    "Company announced quarterly earnings report",
    "Market closed higher today with gains in IT sector",
    "Mutual funds are subject to market risk",
    "Diversify portfolio for long term stability"
]

X = fraudulent_msgs + safe_msgs
y = [1]*len(fraudulent_msgs) + [0]*len(safe_msgs)

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)
clf = LogisticRegression()
clf.fit(X_vec, y)

# -------------------- Keyword Rules --------------------
suspicious_keywords = [
    "guaranteed", "sure shot", "100% return", "no risk", "risk free",
    "profit assured", "double money", "pump", "jackpot", "target hit",
    "multibagger", "join fast", "limited time", "exclusive call",
    "get rich quick", "inside info", "next big stock"
]

def fraud_score_keywords(message: str) -> int:
    msg = message.lower()
    return sum(1 for keyword in suspicious_keywords if keyword in msg)

def analyze_message_ml(message: str):
    msg_vec = vectorizer.transform([message])
    prob = clf.predict_proba(msg_vec)[0][1]
    return round(prob * 100, 2)

def hybrid_analysis(message: str):
    keyword_score = fraud_score_keywords(message)
    ml_prob = analyze_message_ml(message)
    final_prob = (ml_prob + keyword_score * 15) / (1 + keyword_score * 0.2)

    if final_prob > 80:
        label = "🚨 Alert: Highly suspicious!"
    elif final_prob > 40:
        label = "⚠️ Caution: Possibly fraudulent"
    else:
        label = "✅ Safe"
    return label, final_prob

# -------------------- Bot Handlers --------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hi! I’m the Fraud Detector Bot with ML + Database.\n\n"
        "👉 Forward me a message to analyze.\n"
        "👉 Use /checkgroup to scan a group.\n"
        "👉 Use /history to see fraud history of this group."
    )

async def check_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    if user_text:
        label, prob = hybrid_analysis(user_text)
        await update.message.reply_text(f"{label}\nFraud Probability: {prob:.2f}%")

# Group-level analysis
async def check_group(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    messages = []
    
    async for msg in context.bot.get_chat_history(chat.id, limit=50):
        if msg.text:
            messages.append(msg.text)

    fraud_count = 0
    for text in messages:
        _, prob = hybrid_analysis(text)
        if prob > 50:
            fraud_count += 1

    fraud_ratio = (fraud_count / len(messages)) * 100 if messages else 0

    if fraud_ratio > 40:
        verdict = "🚨 Group looks Fictitious / Scam-prone!"
    elif fraud_ratio > 20:
        verdict = "⚠️ Group shows some suspicious activity."
    else:
        verdict = "✅ Group looks safe."

    # Save results to Supabase
    supabase.table("group_analysis").insert({
        "group_id": str(chat.id),
        "group_name": chat.title or "Unknown",
        "fraud_ratio": fraud_ratio,
        "suspicious_count": fraud_count,
        "messages_analyzed": len(messages)
    }).execute()

    await update.message.reply_text(
        f"📊 Group Analysis for *{chat.title or 'this group'}*:\n\n"
        f"Messages Analyzed: {len(messages)}\n"
        f"Suspicious Messages: {fraud_count}\n"
        f"Fraud Ratio: {fraud_ratio:.2f}%\n\n"
        f"{verdict}",
        parse_mode="Markdown"
    )

# Fraud history (from Supabase)
async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    response = supabase.table("group_analysis").select("*").eq("group_id", str(chat.id)).order("analyzed_at", desc=True).limit(5).execute()

    if not response.data:
        await update.message.reply_text("📭 No history found for this group yet. Try running /checkgroup first.")
        return

    history_text = f"📜 Fraud History for *{chat.title or 'this group'}*:\n\n"
    for row in response.data:
        history_text += (
            f"- {row['analyzed_at'][:19]} → Fraud Ratio: {row['fraud_ratio']:.2f}% "
            f"(Suspicious: {row['suspicious_count']}/{row['messages_analyzed']})\n"
        )

    await update.message.reply_text(history_text, parse_mode="Markdown")

# -------------------- Main --------------------
def main():
    TOKEN = "8403500260:AAEe8T6MMZyqgWe4la-rfMuEP7eBy01wDNw"
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("checkgroup", check_group))
    app.add_handler(CommandHandler("history", history))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, check_message))

    print("🚀 Bot with Supabase is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
