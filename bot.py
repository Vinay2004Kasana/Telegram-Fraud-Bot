import asyncio
import re
import tempfile
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from config import supabase, TELEGRAM_BOT_TOKEN

# OCR + File parsers
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import docx


# -------------------- Fetch Keywords --------------------
async def get_suspicious_keywords():
    data = supabase.table("suspicious_keywords").select("keyword").execute()
    if data.data:
        return [row["keyword"].lower() for row in data.data]
    return []


# -------------------- Fraud Scoring --------------------
async def fraud_score(message: str):
    msg = message.lower()
    words = re.findall(r"\b\w+\b", msg)

    keywords = await get_suspicious_keywords()
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


# -------------------- Log to Supabase --------------------
async def log_to_supabase(group_id, group_name, sender, message, fraud_score, matched):
    supabase.table("fraud_logs").insert({
        "group_id": str(group_id),
        "group_name": group_name,
        "sender": sender,
        "message": message[:5000],  # avoid oversize
        "fraud_score": fraud_score,
        "matched_keywords": matched
    }).execute()


# -------------------- Analyze Message --------------------
async def analyze_message(message: str, group_id=None, group_name=None, sender=None):
    score, matched = await fraud_score(message)
    await log_to_supabase(group_id, group_name, sender, message, score, matched)

    if score == 0:
        return "✅ Safe: No suspicious content found"
    elif score <= 2:
        return f"⚠️ Caution: Fraud Score {score}. Suspicious terms: {matched}"
    else:
        return f"🚨 Alert: Fraud Score {score}. Highly suspicious!\nMatched terms: {matched}"


# -------------------- Handlers --------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hi! I am the Fraud Detector Bot.\n\n"
        "I analyze messages, images, and docs for stock market scams.\n"
        "Send me any content!"
    )


async def auto_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    if user_text:
        result = await analyze_message(
            user_text,
            group_id=update.effective_chat.id,
            group_name=update.effective_chat.title or "Private Chat",
            sender=update.effective_user.username or str(update.effective_user.id)
        )
        await update.message.reply_text(result)


async def check_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await update.message.photo[-1].get_file()
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        await file.download_to_drive(tmp.name)
        text = pytesseract.image_to_string(Image.open(tmp.name))

    if text.strip():
        result = await analyze_message(
            text,
            group_id=update.effective_chat.id,
            group_name=update.effective_chat.title or "Private Chat",
            sender=update.effective_user.username or str(update.effective_user.id)
        )
        await update.message.reply_text(f"📸 Extracted text from image:\n\n{text[:200]}...\n\nResult:\n{result}")
    else:
        await update.message.reply_text("⚠️ No readable text found in this image.")


async def check_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await update.message.document.get_file()
    suffix = update.message.document.file_name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(suffix=f".{suffix}") as tmp:
        await file.download_to_drive(tmp.name)
        text = ""

        if suffix == "pdf":
            doc = fitz.open(tmp.name)
            for page in doc:
                text += page.get_text()
        elif suffix in ["docx", "doc"]:
            d = docx.Document(tmp.name)
            text = "\n".join([p.text for p in d.paragraphs])
        elif suffix == "txt":
            text = open(tmp.name, "r", encoding="utf-8", errors="ignore").read()

    if text.strip():
        result = await analyze_message(
            text,
            group_id=update.effective_chat.id,
            group_name=update.effective_chat.title or "Private Chat",
            sender=update.effective_user.username or str(update.effective_user.id)
        )
        await update.message.reply_text(f"📄 Extracted text from doc:\n\n{text[:200]}...\n\nResult:\n{result}")
    else:
        await update.message.reply_text("⚠️ No readable text found in this document.")


# -------------------- Main --------------------
async def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, auto_check))
    app.add_handler(MessageHandler(filters.PHOTO, check_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, check_document))

    print("🚀 Fraud Detector Bot is running...")
    await app.run_polling()


if __name__ == "__main__":
    asyncio.run(main())
