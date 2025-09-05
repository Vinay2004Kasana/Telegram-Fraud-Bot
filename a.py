import streamlit as st

# Suspicious keywords/phrases
suspicious_keywords = [
    "guaranteed", "100% return", "sure shot", "double money",
    "buy call", "sell call", "jackpot", "pump", "target hit",
    "no loss", "risk free", "insider tip", "multibagger",
    "next big stock", "join fast", "limited time", "profit assured"
]

def fraud_score(message: str) -> int:
    msg = message.lower()
    score = sum(keyword in msg for keyword in suspicious_keywords)
    return score

def analyze_message(message: str):
    score = fraud_score(message)
    if score == 0:
        return "✅ Safe: No suspicious content found", "green"
    elif score <= 2:
        return f"⚠️ Caution: Fraud Score {score}. Contains suspicious terms.", "orange"
    else:
        return f"🚨 Alert: Fraud Score {score}. Highly suspicious message!", "red"

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Fraud Message Detector", page_icon="🕵️", layout="centered")

st.title("🕵️ Fraudulent Investment Message Detector")
st.write("Paste any Telegram/WhatsApp/Stock Tip message below to check if it's **fraudulent or safe**.")

user_input = st.text_area("Paste message here:")

if st.button("Check Message"):
    if user_input.strip():
        result, color = analyze_message(user_input)
        st.markdown(f"<h3 style='color:{color};'>{result}</h3>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a message to analyze.")
