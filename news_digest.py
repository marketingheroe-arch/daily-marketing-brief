"""
Daily Marketing News Digest
Fetches industry news, platform updates, and competitor moves for each client.
Outputs a JSON file consumed by the Cowork dashboard task.
Optionally uses Gemini API for intelligent summarization.

Usage:
    python news_digest.py                    # Without AI summarization
    GEMINI_API_KEY=xxx python news_digest.py # With Gemini summarization
"""

import json
import os
import sys
import re
import urllib.request
import urllib.parse
import ssl
from xml.etree import ElementTree
from datetime import datetime, timedelta
from html import unescape
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
GMAIL_FROM = os.environ.get("GMAIL_FROM", CONFIG["email"]["to"])
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SSL_CTX = ssl.create_default_context()


def fetch_url(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 DailyDigestBot/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [WARN] Could not fetch {url}: {e}")
        return ""


def parse_google_news_rss(xml_text, max_items=5):
    items = []
    if not xml_text:
        return items
    try:
        root = ElementTree.fromstring(xml_text)
        for item in root.iter("item"):
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            pub_date = item.findtext("pubDate", "")
            source = item.findtext("source", "")
            title = unescape(re.sub(r"<[^>]+>", "", title))
            items.append({"title": title, "link": link, "published": pub_date, "source": source})
            if len(items) >= max_items:
                break
    except ElementTree.ParseError as e:
        print(f"  [WARN] RSS parse error: {e}")
    return items


def build_google_news_url(query, lang="en", country="US", recent_hours=28):
    encoded = urllib.parse.quote(query + f" when:{recent_hours}h")
    return f"https://news.google.com/rss/search?q={encoded}&hl={lang}&gl={country}&ceid={country}:{lang}"


def fetch_client_news(client):
    print(f"\nFetching news for {client['name']}...")
    result = {
        "client": client["name"],
        "vertical": client["vertical"],
        "market": client["market"],
        "industry_news": [],
        "competitor_news": [],
        "platform_updates": []
    }

    for kw in client.get("search_keywords", []):
        lang = "es" if client["market"] == "Spain" else "en"
        country = "ES" if client["market"] == "Spain" else "US"
        url = build_google_news_url(kw, lang=lang, country=country)
        xml = fetch_url(url)
        articles = parse_google_news_rss(xml, max_items=3)
        for a in articles:
            a["keyword"] = kw
        result["industry_news"].extend(articles)
        print(f"  - '{kw}': {len(articles)} articles")

    for comp in client.get("competitors", []):
        url = build_google_news_url(comp, recent_hours=48)
        xml = fetch_url(url)
        articles = parse_google_news_rss(xml, max_items=3)
        for a in articles:
            a["competitor"] = comp
        result["competitor_news"].extend(articles)
        print(f"  - Competitor '{comp}': {len(articles)} articles")

    for platform in client.get("platforms", []):
        feed_config = CONFIG["platform_feeds"].get(platform)
        if feed_config and "blog_rss" in feed_config:
            xml = fetch_url(feed_config["blog_rss"])
            articles = parse_google_news_rss(xml, max_items=3)
            for a in articles:
                a["platform"] = platform
            result["platform_updates"].extend(articles)
            print(f"  - Platform '{platform}': {len(articles)} articles")

    for key in ["industry_news", "competitor_news", "platform_updates"]:
        seen = set()
        deduped = []
        for item in result[key]:
            if item["title"] not in seen:
                seen.add(item["title"])
                deduped.append(item)
        result[key] = deduped

    return result


def summarize_with_gemini(all_news):
    if not GEMINI_API_KEY:
        print("\nNo GEMINI_API_KEY set - skipping AI summarization.")
        return all_news

    print("\nSummarizing with Gemini...")

    for client_news in all_news:
        prompt = f"""You are a senior digital marketing strategist. Analyze these news items for the client "{client_news['client']}" ({client_news['vertical']}, market: {client_news['market']}).

INDUSTRY NEWS:
{json.dumps(client_news['industry_news'][:10], indent=2, ensure_ascii=False)}

COMPETITOR NEWS:
{json.dumps(client_news['competitor_news'][:8], indent=2, ensure_ascii=False)}

PLATFORM UPDATES:
{json.dumps(client_news['platform_updates'][:8], indent=2, ensure_ascii=False)}

Respond in Spanish. Return a JSON object with these keys:
- "top_alerts": Array of 1-3 urgent items that need immediate attention (string descriptions)
- "industry_summary": 2-3 sentence summary of what is happening in their industry
- "competitor_moves": 1-2 sentence summary of competitor activity (or "Sin movimientos relevantes" if none)
- "platform_changes": 1-2 sentence summary of relevant platform updates
- "opportunities": Array of 1-3 actionable opportunities based on the news
- "risk_level": "low" | "medium" | "high" based on how urgent the news is

IMPORTANT: Return ONLY valid JSON, no markdown code blocks."""

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{CONFIG['gemini']['model']}:generateContent?key={GEMINI_API_KEY}"
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1000}
        }).encode("utf-8")

        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                text = body["candidates"][0]["content"]["parts"][0]["text"]
                text = re.sub(r"```json\s*", "", text)
                text = re.sub(r"```\s*", "", text)
                ai_summary = json.loads(text)
                client_news["ai_summary"] = ai_summary
                print(f"  Done: {client_news['client']}: risk={ai_summary.get('risk_level', '?')}")
        except Exception as e:
            print(f"  Gemini error for {client_news['client']}: {e}")
            client_news["ai_summary"] = None

    return all_news


def send_email_digest(all_news, date_str):
    if not GMAIL_APP_PASSWORD:
        print("\nNo GMAIL_APP_PASSWORD set - skipping email.")
        return

    print("\nSending email digest...")

    body_parts = [f"MORNING BRIEF - {date_str}\n{'='*50}\n"]

    for client_news in all_news:
        body_parts.append(f"\n\n{'='*50}")
        body_parts.append(f"{client_news['client'].upper()} ({client_news['vertical']})")
        body_parts.append(f"{'='*50}")

        if client_news.get("ai_summary"):
            ai = client_news["ai_summary"]
            if ai.get("top_alerts"):
                body_parts.append(f"\nALERTAS:")
                for alert in ai["top_alerts"]:
                    body_parts.append(f"  - {alert}")
            body_parts.append(f"\nIndustria: {ai.get('industry_summary', 'N/A')}")
            body_parts.append(f"Competencia: {ai.get('competitor_moves', 'N/A')}")
            body_parts.append(f"Plataformas: {ai.get('platform_changes', 'N/A')}")
            if ai.get("opportunities"):
                body_parts.append(f"\nOPORTUNIDADES:")
                for opp in ai["opportunities"]:
                    body_parts.append(f"  - {opp}")
        else:
            body_parts.append(f"\nNoticias de industria ({len(client_news['industry_news'])} articulos):")
            for item in client_news["industry_news"][:5]:
                body_parts.append(f"  - {item['title']}")
                body_parts.append(f"    {item['link']}")

            if client_news["competitor_news"]:
                body_parts.append(f"\nCompetencia ({len(client_news['competitor_news'])} articulos):")
                for item in client_news["competitor_news"][:3]:
                    body_parts.append(f"  - [{item.get('competitor','')}] {item['title']}")

            if client_news["platform_updates"]:
                body_parts.append(f"\nPlataformas ({len(client_news['platform_updates'])} articulos):")
                for item in client_news["platform_updates"][:3]:
                    body_parts.append(f"  - [{item.get('platform','')}] {item['title']}")

    body = "\n".join(body_parts)

    msg = MIMEMultipart()
    msg["From"] = GMAIL_FROM
    msg["To"] = CONFIG["email"]["to"]
    msg["Subject"] = f"{CONFIG['email']['subject_prefix']} - {date_str}"
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_FROM, GMAIL_APP_PASSWORD)
            server.send_message(msg)
        print("  Email sent successfully!")
    except Exception as e:
        print(f"  Email error: {e}")


def main():
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"Daily Marketing Brief - {today}")
    print(f"   Clients: {len(CONFIG['clients'])}")
    print(f"   Gemini: {'enabled' if GEMINI_API_KEY else 'disabled'}")
    print(f"   Email: {'enabled' if GMAIL_APP_PASSWORD else 'disabled'}")

    all_news = []
    for client in CONFIG["clients"]:
        news = fetch_client_news(client)
        all_news.append(news)

    if CONFIG["gemini"]["enabled"] and GEMINI_API_KEY:
        all_news = summarize_with_gemini(all_news)

    output = {
        "date": today,
        "generated_at": datetime.now().isoformat(),
        "clients": all_news
    }
    output_path = os.path.join(OUTPUT_DIR, "latest_digest.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")

    archive_path = os.path.join(OUTPUT_DIR, f"digest_{today}.json")
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    send_email_digest(all_news, today)

    print(f"\nDone! Processed {len(all_news)} clients.")


if __name__ == "__main__":
    main()
