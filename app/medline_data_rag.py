
import requests
import xml.etree.ElementTree as ET
import pandas as pd

from bs4 import BeautifulSoup
from .app_logging import logger
from datetime import datetime, timedelta

today = datetime.now()
yesterday = today - timedelta(days=1)
data_str = yesterday.strftime("%Y-%m-%d")

XML_URL = f"https://medlineplus.gov/xml/mplus_topics_{data_str}.xml"
RAG_FILE = "./../medlineplus_knowledge_base.csv"


def clean_html_text(html_content):
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def download_and_process():
    logger.info(f"[INFO] Starting download from MedlinePlus dataset: {XML_URL}...")

    response = requests.get(XML_URL)

    if response.status_code != 200:
        logger.error(f"[ERROR] Download error! Status: {response.status_code}")
        return

    root = ET.fromstring(response.content)

    data_rows = []

    for topic in root.findall('health-topic'):
        url = topic.get('url')
        if url and "/spanish/" in url:
            continue

        title = topic.get('title')
        mplus_id = topic.get('id')

        summary_tag = topic.find('full-summary')
        raw_summary = summary_tag.text if summary_tag is not None else ""
        clean_summary = clean_html_text(raw_summary)

        data_rows.append({
            "id": mplus_id,
            "title": title,
            "description": clean_summary,
            "source_url": url,
            "source_name": "MedlinePlus"
        })

    df = pd.DataFrame(data_rows)
    df.to_csv(RAG_FILE, index=False, encoding='utf-8')

    logger.info(f"[INFO] Data written to file: {RAG_FILE}. Processed {len(df)} records.")
