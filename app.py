"""
app.py - Email Client Security Dashboard
-----------------------------------------
Hybrid Spam & Phishing Detection System
Interactive email client dashboard featuring st.session_state mailbox management,
sidebar folder navigation (Inbox vs Spam Folder), automated incoming email simulation,
and multi-tier threat classification (Cybersecurity URL Inspector + Naive Bayes AI).
"""

import os
import uuid
import string
import joblib
import nltk
from nltk.corpus import stopwords
from datetime import datetime
import streamlit as st
from url_inspector import check_url_security

# Page configuration
st.set_page_config(
    page_title="SecureMail - Hybrid Threat Shield",
    page_icon="📬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern corporate webmail client
st.markdown("""
<style>
    .client-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.1rem;
    }
    .client-sub {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .folder-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .email-meta-header {
        background-color: #F8FAFC;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
        border: 1px solid #E2E8F0;
    }
    .threat-tag {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .tag-cyber {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
    }
    .tag-ai {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
    }
    .tag-safe {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
    }
</style>
""", unsafe_allow_html=True)

MODEL_FILE = "spam_model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"


@st.cache_resource(show_spinner="Initializing Security ML Models...")
def load_ml_resources():
    """Loads NLTK stopwords, trained Naive Bayes classifier, and TF-IDF vectorizer."""
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))

    model = None
    vectorizer = None
    ready = os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE)

    if ready:
        try:
            model = joblib.load(MODEL_FILE)
            vectorizer = joblib.load(VECTORIZER_FILE)
        except Exception as exc:
            st.error(f"Error loading model artifacts: {exc}")

    return model, vectorizer, stop_words, ready


def preprocess_text(text: str, stop_words: set) -> str:
    """Cleans input text identically to the training pipeline."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)


def classify_email(email_body: str, model, vectorizer, stop_words):
    """
    Two-stage hybrid classification pipeline:
    1. Cybersecurity URL Inspector (domain age < 30 days) -> High-Risk Phishing
    2. Machine Learning NLP Engine (TF-IDF + Multinomial Naive Bayes) -> Spam vs Ham
    """
    # ----------------- STAGE 1: URL INSPECTION -----------------
    url_check = check_url_security(email_body)

    if url_check.get("is_high_risk", False):
        return {
            "status": "Spam",
            "caught_by": "Cybersecurity (URL Inspector)",
            "threat_reason": f"Suspicious Domain Age (<30 days): {url_check.get('message')}",
            "threat_type": "Zero-Day Phishing",
            "details": url_check
        }

    # ----------------- STAGE 2: MACHINE LEARNING NLP -----------------
    if model is not None and vectorizer is not None:
        cleaned = preprocess_text(email_body, stop_words)
        vec_features = vectorizer.transform([cleaned])
        prediction = model.predict(vec_features)[0]
        probabilities = model.predict_proba(vec_features)[0]

        classes = list(model.classes_)
        spam_idx = classes.index("spam") if "spam" in classes else 1
        ham_idx = classes.index("ham") if "ham" in classes else 0

        spam_prob = probabilities[spam_idx] * 100
        ham_prob = probabilities[ham_idx] * 100

        if prediction.lower() == "spam":
            return {
                "status": "Spam",
                "caught_by": "ML Engine (Naive Bayes AI)",
                "threat_reason": f"Flagged by AI: Linguistic features match promotional/fraudulent spam patterns ({spam_prob:.1f}% confidence)",
                "threat_type": "Textual Spam",
                "confidence": spam_prob
            }
        else:
            return {
                "status": "Safe",
                "caught_by": None,
                "threat_reason": None,
                "threat_type": "None",
                "confidence": ham_prob
            }

    # Fallback if model missing
    return {
        "status": "Safe",
        "caught_by": None,
        "threat_reason": None,
        "threat_type": "None",
        "confidence": 100.0
    }


# Load models
model, vectorizer, stop_words, models_ready = load_ml_resources()

# ----------------- SESSION STATE INITIALIZATION -----------------
# Stores array of email dictionaries: id, sender, subject, body, status, threat_reason, caught_by, timestamp
if "emails" not in st.session_state:
    st.session_state.emails = []
    # Seed with initial default emails
    initial_seed = [
        {
            "id": "init-1",
            "sender": "sarah.connor@cyberdyne-tech.io",
            "subject": "Sprint 14 Retrospective & Q3 Architecture Planning",
            "body": "Hi team,\n\nAttached are the slide decks and database migration diagrams for our upcoming sprint review tomorrow at 10:00 AM EST. Please review the schema changes and add any backlog items before the sync.\n\nBest regards,\nSarah Connor",
            "status": "Safe",
            "threat_reason": None,
            "caught_by": None,
            "timestamp": "Today, 09:15 AM"
        },
        {
            "id": "init-2",
            "sender": "rewards-promo@claim-voucher-dept.net",
            "subject": "CONGRATULATIONS!! You won a £1,000 Gift Voucher!",
            "body": "URGENT WINNER NOTICE!! You have been selected as our lucky mobile user to receive a guaranteed £1,000 cash prize voucher reward! Call 09061701461 right now to claim code KL341 before midnight! Reply STOP to cancel.",
            "status": "Spam",
            "threat_reason": "Flagged by AI: Linguistic features match promotional/fraudulent spam patterns (98.4% confidence)",
            "caught_by": "ML Engine (Naive Bayes AI)",
            "timestamp": "Today, 10:42 AM"
        },
        {
            "id": "init-3",
            "sender": "it-support@microsoft-365-helpdesk.com",
            "subject": "CRITICAL: Corporate SSO Password Expired - Verify Immediately",
            "body": "SECURITY ALERT: Multiple unauthorized login attempts were detected from an unknown IP address. Your corporate single sign-on access is scheduled for termination within 2 hours. Confirm your credentials immediately at http://new-phish-bank-login-2026.xyz/verify to prevent permanent data deletion.",
            "status": "Spam",
            "threat_reason": "Suspicious Domain Age (<30 days): Phishing Alert: High-risk URL detected! Domain 'new-phish-bank-login-2026.xyz' is newly registered (4 days old < 30 days).",
            "caught_by": "Cybersecurity (URL Inspector)",
            "timestamp": "Today, 11:05 AM"
        }
    ]
    st.session_state.emails = initial_seed

# ----------------- SIMULATE INCOMING EMAILS FUNCTION -----------------
MOCK_TEMPLATES = [
    {
        "sender": "alex.morgan@university.edu",
        "subject": "Final Project Guidelines & Submission Portal Link",
        "body": "Hello everyone,\n\nPlease make sure to submit your final project reports before Friday midnight. The guidelines and standard submission format are outlined on our official documentation page: https://docs.python.org/3/tutorial/. Good luck!\n\nProf. Morgan",
    },
    {
        "sender": "urgent-claims@lottery-bonus-direct.com",
        "subject": "FINAL REMINDER: Claim your £1,500 Award Before Expiration",
        "body": "FINAL NOTICE!! As a premier mobile customer you are entitled to a £1,500 cash payout reward. Call customer claim helpline 09061701461 immediately quoting customer ID TC2026. Do not miss this exclusive prize reward!",
    },
    {
        "sender": "security-team@internal-helpdesk-login.net",
        "subject": "URGENT: Payroll Account Suspension Notice",
        "body": "Dear Employee,\n\nWe encountered a mismatch in your direct deposit bank information. Your bi-weekly salary disbursement is currently on hold. Please update your banking profile immediately at http://fake-bank-login-update.com/dispute to avoid payment interruption.",
    },
    {
        "sender": "jessica.lee@startup-team.co",
        "subject": "Design Review & Wireframe Feedback for Mobile App",
        "body": "Hi team,\n\nI reviewed the latest Figma prototypes for the mobile dashboard. The user flow looks clean, but we should make sure the accessibility contrast ratios pass WCAG standards. Let us discuss this during tomorrow morning standup.\n\nCheers,\nJessica",
    }
]


def simulate_incoming_emails():
    """Generates mock emails, runs them through the hybrid pipeline, and appends to session state."""
    current_time_str = datetime.now().strftime("%I:%M %p")
    new_safe_count = 0
    new_spam_count = 0

    for item in MOCK_TEMPLATES:
        email_id = str(uuid.uuid4())[:8]
        classification = classify_email(item["body"], model, vectorizer, stop_words)

        new_email = {
            "id": email_id,
            "sender": item["sender"],
            "subject": item["subject"],
            "body": item["body"],
            "status": classification["status"],
            "threat_reason": classification["threat_reason"],
            "caught_by": classification["caught_by"],
            "timestamp": f"Just now ({current_time_str})"
        }

        # Prepend to display newest first
        st.session_state.emails.insert(0, new_email)

        if classification["status"] == "Safe":
            new_safe_count += 1
        else:
            new_spam_count += 1

    return new_safe_count, new_spam_count


# ----------------- SIDEBAR CONTROLS & NAVIGATION -----------------
safe_emails = [e for e in st.session_state.emails if e["status"] == "Safe"]
spam_emails = [e for e in st.session_state.emails if e["status"] == "Spam"]

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=64)
    st.markdown("## 🛡️ SecureMail Client")
    st.caption("Hybrid ML + Zero-Day Cybersecurity Defense")

    st.divider()

    # Simulation Button
    st.markdown("### ⚡ Live Mailbox Simulator")
    if st.button("Simulate Incoming Emails", type="primary", use_container_width=True):
        safe_cnt, spam_cnt = simulate_incoming_emails()
        st.success(f"📥 Processed {safe_cnt + spam_cnt} new emails:\n- {safe_cnt} routed to Inbox\n- {spam_cnt} quarantined to Spam")
        st.rerun()

    st.divider()

    # Sidebar Navigation Tabs
    st.markdown("### 📁 Mailbox Folders")
    nav_selection = st.radio(
        label="Select Folder:",
        options=[
            f"📥 Inbox ({len(safe_emails)})",
            f"🚨 Spam Folder ({len(spam_emails)})"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    # System Status
    st.markdown("### 🧩 Engine Status")
    if models_ready:
        st.success("✅ Naive Bayes AI: Active")
        st.success("✅ URL WHOIS Inspector: Active")
    else:
        st.error("❌ Model artifacts missing! Run `train_model.py`.")

    st.caption("Cybersecurity Policy: Domains < 30 days old are quarantined automatically.")


# ----------------- MAIN SCREEN CONTENT -----------------
st.markdown('<div class="client-header">📬 Corporate Email Security Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="client-sub">Multi-stage inspection protecting inbound communications from phishing and spam</div>', unsafe_allow_html=True)

# Top Metrics Row
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Total Inbound Messages", len(st.session_state.emails))
with col_m2:
    st.metric("Clean Messages (Inbox)", len(safe_emails), delta="Safe", delta_color="normal")
with col_m3:
    st.metric("Quarantined Messages (Spam)", len(spam_emails), delta="Blocked", delta_color="inverse")
with col_m4:
    url_quarantined = len([e for e in spam_emails if e.get("caught_by") == "Cybersecurity (URL Inspector)"])
    st.metric("Zero-Day Phishing Intercepted", url_quarantined, delta="High-Risk", delta_color="inverse")

st.divider()

# ----------------- INBOX VIEW -----------------
if "Inbox" in nav_selection:
    col_ih1, col_ih2 = st.columns([4, 1])
    with col_ih1:
        st.markdown(f'<div class="folder-title">📥 Inbox — Clean & Verified Communications ({len(safe_emails)})</div>', unsafe_allow_html=True)
    with col_ih2:
        if safe_emails:
            if st.button("Clear All Safe Mails", key="clear_all_safe", use_container_width=True):
                st.session_state.emails = [e for e in st.session_state.emails if e["status"] != "Safe"]
                st.rerun()

    if not safe_emails:
        st.info("ℹ️ Your Inbox is currently empty. Click **'Simulate Incoming Emails'** in the sidebar to fetch incoming messages.")
    else:
        for idx, email in enumerate(safe_emails):
            expander_title = f"✉️  {email['subject']}   |   From: {email['sender']}   •   {email.get('timestamp', '')}"
            with st.expander(expander_title, expanded=(idx == 0)):
                # Metadata Box
                st.markdown(f"""
                <div class="email-meta-header">
                    <strong>From:</strong> {email['sender']}<br>
                    <strong>Subject:</strong> {email['subject']}<br>
                    <strong>Received:</strong> {email.get('timestamp', '')} &nbsp;|&nbsp;
                    <span class="threat-tag tag-safe">VERIFIED SAFE</span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Message Content:**")
                st.write(email["body"])

                st.divider()

                col_act1, col_act2, col_act3 = st.columns([2, 2, 6])
                with col_act1:
                    # Move to spam manually
                    if st.button("⚠️ Mark as Spam", key=f"mark_spam_{email['id']}"):
                        for e in st.session_state.emails:
                            if e["id"] == email["id"]:
                                e["status"] = "Spam"
                                e["threat_reason"] = "Manually flagged as Spam by user"
                                e["caught_by"] = "User Manual Action"
                        st.rerun()
                with col_act2:
                    # Delete email
                    if st.button("🗑️ Delete", key=f"del_inbox_{email['id']}"):
                        st.session_state.emails = [e for e in st.session_state.emails if e["id"] != email["id"]]
                        st.rerun()

# ----------------- SPAM FOLDER VIEW -----------------
elif "Spam Folder" in nav_selection:
    col_sh1, col_sh2 = st.columns([4, 1])
    with col_sh1:
        st.markdown(f'<div class="folder-title">🚨 Spam & Phishing Quarantine Folder ({len(spam_emails)})</div>', unsafe_allow_html=True)
    with col_sh2:
        if spam_emails:
            if st.button("🗑️ Empty Spam Folder", type="primary", key="empty_spam", use_container_width=True):
                st.session_state.emails = [e for e in st.session_state.emails if e["status"] != "Spam"]
                st.rerun()

    if not spam_emails:
        st.success("🎉 No spam or phishing threats quarantined. All clean!")
    else:
        st.warning("⚠️ **Security Warning:** The following emails were quarantined by the hybrid detection system. Links and scripts have been neutralized.")

        for idx, email in enumerate(spam_emails):
            caught_by_tag = email.get("caught_by", "Threat Filter")
            expander_title = f"🚨  {email['subject']}   |   From: {email['sender']}   [{caught_by_tag}]"

            with st.expander(expander_title, expanded=(idx == 0)):
                # Clear colored warning banner showing reason
                caught_by_lower = caught_by_tag.lower()
                tag_class = "tag-cyber" if "url" in caught_by_lower or "cyber" in caught_by_lower else "tag-ai"

                st.error(f"**🛡️ Quarantined Threat Assessment:**\n\n- **Detection Mechanism:** `{caught_by_tag}`\n- **Threat Reason:** {email.get('threat_reason', 'Flagged as high-risk by security filter')}")

                st.markdown(f"""
                <div class="email-meta-header" style="background-color: #FEF2F2; border-color: #FECACA;">
                    <strong>From:</strong> {email['sender']}<br>
                    <strong>Subject:</strong> {email['subject']}<br>
                    <strong>Timestamp:</strong> {email.get('timestamp', '')} &nbsp;|&nbsp;
                    <span class="threat-tag {tag_class}">QUARANTINED</span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Quarantined Email Body (Safe Preview):**")
                # Render in text area to neutralize any accidental clicks on malicious links
                st.text_area(
                    label="Neutralized Content Preview:",
                    value=email["body"],
                    height=130,
                    disabled=True,
                    key=f"preview_body_{email['id']}"
                )

                st.divider()

                col_sact1, col_sact2, col_sact3 = st.columns([2, 2, 6])
                with col_sact1:
                    # Delete forever button
                    if st.button("🗑️ Delete Forever", type="primary", key=f"del_forever_{email['id']}"):
                        st.session_state.emails = [e for e in st.session_state.emails if e["id"] != email["id"]]
                        st.rerun()
                with col_sact2:
                    # Restore to Inbox
                    if st.button("↩️ Move to Inbox", key=f"restore_{email['id']}"):
                        for e in st.session_state.emails:
                            if e["id"] == email["id"]:
                                e["status"] = "Safe"
                                e["threat_reason"] = None
                                e["caught_by"] = None
                        st.rerun()

st.divider()
st.caption("SecureMail Hybrid Client | Built with Streamlit, Python WHOIS Heuristics & Scikit-Learn Multinomial Naive Bayes")
