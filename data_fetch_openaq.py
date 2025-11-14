# data_fetch_openaq.py (HARDCODED API KEY VERSION)
# You do NOT need to set environment variables.

import requests, time
from pathlib import Path
import pandas as pd

API_KEY = "9caebbe8e15fe35557f5842bddeb451948a422a5dc2efb27eaab143e7a87c00b"  # <-- Your key here

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

CITIES = [
    "Chennai","Coimbatore","Madurai","Tiruchirappalli","Salem",
    "Vellore","Tirunelveli","Erode","Thoothukudi"
]

BASE = "https://api.openaq.org/v3/measurements"
LIMIT = 100
SLEEP_BETWEEN_PAGES = 1.0
TIMEOUT = 30

HEADERS = {"X-API-Key": API_KEY}

def fetch_city(city):
    print(f"\nFetching city: {city}")
    page = 1
    all_rows = []

    while True:
        params = {
            "city": city,
            "limit": LIMIT,
            "page": page,
            "sort": "desc",
            "order_by": "date"
        }

        r = requests.get(BASE, headers=HEADERS, params=params, timeout=TIMEOUT)

        if r.status_code == 401:
            print("  ERROR 401: Invalid API key.")
            return []
        if r.status_code != 200:
            print(f"  HTTP {r.status_code}: {r.text[:200]}")
            return []

        payload = r.json()
        results = payload.get("results") or []
        meta = payload.get("meta") or {}

        if not results:
            break

        all_rows.extend(results)

        total_pages = meta.get("totalPages")
        if total_pages and page >= total_pages:
            break

        page += 1
        time.sleep(SLEEP_BETWEEN_PAGES)

    print(f"  Collected {len(all_rows)} rows for {city}")
    return all_rows

def save_city(city, rows):
    if not rows:
        print(f"  No data for {city}")
        return
    df = pd.json_normalize(rows)
    df.to_csv(OUT / f"{city.replace(' ','_')}_openaq_v3.csv", index=False)
    print(f"  Saved -> {city.replace(' ','_')}_openaq_v3.csv")

if __name__ == "__main__":
    for city in CITIES:
        rows = fetch_city(city)
        save_city(city, rows)
    print("\nDone! Check data/raw/")
