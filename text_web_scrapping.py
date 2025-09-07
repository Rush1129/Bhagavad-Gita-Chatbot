import requests
from bs4 import BeautifulSoup
import json
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

base_url = "https://vedabase.io"
all_data = {}

def get_verse_details(url):
    """Fetch Sanskrit, translation, and purport from a verse page."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Sanskrit text
    sanskrit_div = soup.select_one("div.em\\:mb-4.em\\:leading-8.em\\:text-lg.text-center")
    sanskrit_text = sanskrit_div.get_text(separator=" ", strip=True) if sanskrit_div else None

    # Translation
    translation_div = soup.select_one("div.em\\:mb-4.em\\:leading-8.em\\:text-base.s-justify strong")
    translation_text = translation_div.get_text(strip=True) if translation_div else None

    # Purport
    purport_div = soup.select_one("div.av-purport")
    purport_text = ""
    if purport_div:
        paragraphs = purport_div.find_all("div", class_="em:mb-4 em:leading-8 em:text-base s-justify")
        purport_text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)

    return {
        "sanskrit": sanskrit_text,
        "translation": translation_text,
        "purport": purport_text
    }

# Loop through chapters 1 to 18
for chapter in range(1, 19):
    print(f"Scraping Chapter {chapter}...")
    chapter_url = f"{base_url}/en/library/bg/{chapter}/"
    response = requests.get(chapter_url)
    soup = BeautifulSoup(response.text, "html.parser")

    chapter_data = {}
    verse_links = []

    # Extract all verse links
    for div in soup.find_all("div", class_="em:mb-4 em:leading-8 em:text-base text-justify"):
        a_tag = div.find("a", href=True)
        if a_tag and f"/en/library/bg/{chapter}/" in a_tag['href']:
            verse_text = a_tag.get_text(strip=True).replace("TEXT ", "").replace(":", "").replace("TEXTS ", "")  # e.g., "1" or "16-18"
            verse_url = base_url + a_tag['href']
            verse_links.append((verse_text, verse_url))

    # Fetch each verse details
    for verse_text, verse_url in verse_links:
        print(f"  Fetching verse {verse_text} from {verse_url}")
        details = get_verse_details(verse_url)

        # Store combined verses (like "16-18") as is
        chapter_data[verse_text] = details

        time.sleep(0.1)  # polite delay

    all_data[f"chapter_{chapter}"] = chapter_data

# Save all data to JSON
with open("bhagavad_gita.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print("Scraping completed. Data saved to bhagavad_gita.json")
