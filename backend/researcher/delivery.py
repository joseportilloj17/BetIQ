"""
delivery.py — Send researcher briefs via email and SMS.

Email:  smtplib (Gmail app password or any SMTP)
SMS:    Free email-to-SMS gateway (carrier@gateway)
        Fallback: Twilio (if configured)

Configuration lives in ../data/delivery_config.json:
{
    "email": {
        "enabled": true,
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "smtp_user": "you@gmail.com",
        "smtp_pass": "xxxx-xxxx-xxxx-xxxx",   // Gmail app password
        "to": "you@gmail.com",
        "mode": "full"                          // "full" or "summary"
    },
    "sms": {
        "enabled": true,
        "method": "gateway",                    // "gateway" or "twilio"
        "gateway_address": "5551234567@vtext.com",
        "twilio_sid": "",
        "twilio_token": "",
        "twilio_from": "",
        "twilio_to": "",
        "mode": "summary"                       // "full" or "summary"
    },
    "send_time_ct": "09:30"                     // HH:MM in CT
}

Gateway addresses:
  Verizon:   number@vtext.com
  AT&T:      number@txt.att.net
  T-Mobile:  number@tmomail.net
  Sprint:    number@messaging.sprintpcs.com
"""
import json
import logging
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

# Load .env file if it exists (for SMTP_PASS and other secrets)
_ENV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"
)
if os.path.exists(_ENV_PATH):
    with open(_ENV_PATH) as _ef:
        for _line in _ef:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "delivery_config.json"
)

# ── SMS character limit (gateway) ────────────────────────────────────────────
_SMS_CHAR_LIMIT = 1500  # multi-segment SMS; gateways vary (160 per segment)


def _load_config() -> dict:
    """Load delivery config from disk."""
    if not os.path.exists(_CONFIG_PATH):
        return {}
    with open(_CONFIG_PATH) as f:
        return json.load(f)


def _save_config(cfg: dict):
    """Write delivery config to disk."""
    os.makedirs(os.path.dirname(_CONFIG_PATH), exist_ok=True)
    with open(_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def get_config() -> dict:
    """Return current delivery config (for API/UI)."""
    return _load_config()


def update_config(updates: dict) -> dict:
    """Merge updates into config and save."""
    cfg = _load_config()
    cfg.update(updates)
    _save_config(cfg)
    return cfg


def _summarize_brief(md_text: str, max_chars: int = _SMS_CHAR_LIMIT) -> str:
    """Extract a short summary from the raw brief markdown for SMS."""
    lines = md_text.split("\n")
    parts = []
    in_injury = False
    injury_count = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# "):
            # Title line — abbreviate
            parts.append(stripped.replace("# BetIQ MLB Raw Brief", "BetIQ Brief"))
            continue
        if stripped.startswith("## Injury"):
            in_injury = True
            continue
        if stripped.startswith("## ") and in_injury:
            in_injury = False
            continue
        if in_injury and stripped.startswith("- **") and injury_count < 5:
            # Extract just the bold headline
            bold_end = stripped.find("**", 3)
            if bold_end > 0:
                parts.append(stripped[2:bold_end + 2])
            injury_count += 1
            continue
        if stripped.startswith("## Today's Slate"):
            # Count games
            game_lines = []
            continue

    # Fallback: first N lines
    if not parts:
        for line in lines[:15]:
            if line.strip():
                parts.append(line.strip())

    summary = "\n".join(parts)
    if len(summary) > max_chars:
        summary = summary[:max_chars - 3] + "..."
    return summary


def send_email(subject: str, body_md: str, cfg: dict = None) -> dict:
    """Send brief via email (SMTP)."""
    cfg = cfg or _load_config()
    email_cfg = cfg.get("email", {})

    if not email_cfg.get("enabled"):
        return {"sent": False, "reason": "email not enabled"}

    smtp_host = email_cfg.get("smtp_host", "smtp.gmail.com")
    smtp_port = email_cfg.get("smtp_port", 587)
    smtp_user = email_cfg.get("smtp_user")
    smtp_pass = email_cfg.get("smtp_pass") or os.environ.get("SMTP_PASS", "")
    to_addr = email_cfg.get("to")
    mode = email_cfg.get("mode", "full")

    if not all([smtp_user, smtp_pass, to_addr]):
        return {"sent": False, "reason": "email credentials not configured"}

    content = body_md if mode == "full" else _summarize_brief(body_md)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_addr
    msg.attach(MIMEText(content, "plain", "utf-8"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        logger.info("[delivery] Email sent to %s", to_addr)
        return {"sent": True, "to": to_addr, "mode": mode, "chars": len(content)}
    except Exception as exc:
        logger.error("[delivery] Email failed: %s", exc)
        return {"sent": False, "error": str(exc)}


def send_sms(body_md: str, cfg: dict = None) -> dict:
    """Send brief via SMS (gateway or Twilio)."""
    cfg = cfg or _load_config()
    sms_cfg = cfg.get("sms", {})

    if not sms_cfg.get("enabled"):
        return {"sent": False, "reason": "sms not enabled"}

    mode = sms_cfg.get("mode", "summary")
    content = _summarize_brief(body_md) if mode == "summary" else body_md[:_SMS_CHAR_LIMIT]

    method = sms_cfg.get("method", "gateway")

    if method == "gateway":
        return _send_sms_gateway(content, sms_cfg, cfg)
    elif method == "twilio":
        return _send_sms_twilio(content, sms_cfg)
    else:
        return {"sent": False, "reason": f"unknown sms method: {method}"}


def _send_sms_gateway(content: str, sms_cfg: dict, cfg: dict) -> dict:
    """Send SMS via email-to-SMS gateway (free)."""
    gateway = sms_cfg.get("gateway_address")
    if not gateway:
        return {"sent": False, "reason": "gateway_address not configured"}

    # Reuse email SMTP settings
    email_cfg = cfg.get("email", {})
    smtp_host = email_cfg.get("smtp_host", "smtp.gmail.com")
    smtp_port = email_cfg.get("smtp_port", 587)
    smtp_user = email_cfg.get("smtp_user")
    smtp_pass = email_cfg.get("smtp_pass") or os.environ.get("SMTP_PASS", "")

    if not all([smtp_user, smtp_pass]):
        return {"sent": False, "reason": "smtp credentials needed for gateway SMS"}

    msg = MIMEText(content, "plain", "utf-8")
    msg["From"] = smtp_user
    msg["To"] = gateway
    msg["Subject"] = "BetIQ Brief"

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        logger.info("[delivery] SMS (gateway) sent to %s", gateway)
        return {"sent": True, "to": gateway, "method": "gateway", "chars": len(content)}
    except Exception as exc:
        logger.error("[delivery] SMS gateway failed: %s", exc)
        return {"sent": False, "error": str(exc)}


def _send_sms_twilio(content: str, sms_cfg: dict) -> dict:
    """Send SMS via Twilio (~1c/msg)."""
    sid = sms_cfg.get("twilio_sid")
    token = sms_cfg.get("twilio_token")
    from_num = sms_cfg.get("twilio_from")
    to_num = sms_cfg.get("twilio_to")

    if not all([sid, token, from_num, to_num]):
        return {"sent": False, "reason": "twilio credentials not configured"}

    try:
        import requests
        url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
        resp = requests.post(url, auth=(sid, token), data={
            "From": from_num,
            "To": to_num,
            "Body": content,
        }, timeout=15)
        if resp.status_code in (200, 201):
            logger.info("[delivery] SMS (twilio) sent to %s", to_num)
            return {"sent": True, "to": to_num, "method": "twilio", "chars": len(content)}
        else:
            return {"sent": False, "error": resp.text[:200]}
    except Exception as exc:
        logger.error("[delivery] Twilio failed: %s", exc)
        return {"sent": False, "error": str(exc)}


def deliver_brief(game_date: str, md_text: str) -> dict:
    """Send brief via all enabled channels. Called after researcher run."""
    cfg = _load_config()
    results = {"game_date": game_date}

    # Email
    subject = f"BetIQ Daily Brief — {game_date}"
    results["email"] = send_email(subject, md_text, cfg)

    # SMS
    results["sms"] = send_sms(md_text, cfg)

    return results
