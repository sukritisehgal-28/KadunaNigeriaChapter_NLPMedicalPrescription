import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

# File names
INPUT_EXCEL = "condition.xlsx"  # Change to your input file
OUTPUT_EXCEL = "medicine_data.xlsx"

def scrape_medicine_data(medicine_name, url):
    """Scrape medicine data from the given URL with detailed logging."""
    print(f"\n[INFO] Processing: {medicine_name} | URL: {url}")
    try:
        print(f"[INFO] Sending request to {url}...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise error for bad response
        print(f"[SUCCESS] Successfully fetched data from {url}")

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract the first heading (h1, h2, h3, etc.)
        heading_tag = soup.find(["h1", "h2", "h3"])
        heading = heading_tag.text.strip() if heading_tag else "No heading found"
        print(f"[INFO] Extracted heading: {heading}")

        # Extract all paragraph contents
        paragraphs = [p.text.strip() for p in soup.find_all("p")]
        content = "\n\n".join(paragraphs) if paragraphs else "No content found"
        print(f"[INFO] Extracted {len(paragraphs)} paragraph(s).")

        return [medicine_name, url, heading, content]
    
    except requests.Timeout:
        print(f"[ERROR] Timeout occurred while accessing {url}")
        return [medicine_name, url, "Error", "Timeout Error"]
    
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch {url} | Reason: {e}")
        return [medicine_name, url, "Error", str(e)]

def main():
    print("[INFO] Starting medicine data scraping...\n")

    # Read Excel file
    try:
        print(f"[INFO] Loading data from {INPUT_EXCEL}...")
        df = pd.read_excel(INPUT_EXCEL)
        print(f"[SUCCESS] Loaded {len(df)} records from {INPUT_EXCEL}.")
    except Exception as e:
        print(f"[ERROR] Failed to load {INPUT_EXCEL} | Reason: {e}")
        return

    # Check if expected columns exist
    if "medicine_name" not in df.columns or "url" not in df.columns:
        print("[ERROR] Excel file must contain 'medicine_name' and 'url' columns.")
        return

    # List to store scraped data
    scraped_data = []

    # Iterate through each row and scrape data
    for index, row in df.iterrows():
        medicine_name = row["medicine_name"]
        url = row["url"]
        scraped_data.append(scrape_medicine_data(medicine_name, url))
        time.sleep(1)  # Adding delay to avoid being blocked by websites

    # Convert to DataFrame and save to Excel
    result_df = pd.DataFrame(scraped_data, columns=["Medicine Name", "URL", "Heading", "Content"])
    result_df.to_excel(OUTPUT_EXCEL, index=False)

    print(f"\n[INFO] Scraping complete! Data saved to {OUTPUT_EXCEL}")
    print("[INFO] Exiting script.\n")

if __name__ == "__main__":
    main()
