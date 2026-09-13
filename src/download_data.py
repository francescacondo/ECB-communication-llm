from pathlib import Path

import requests


ECB_SPEECHES_URL = (
    "https://www.ecb.europa.eu/press/key/shared/data/all_ECB_speeches.csv"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = RAW_DATA_DIR / "all_ECB_speeches.csv"


def download_ecb_speeches() -> None:
    """Download the official ECB speeches dataset."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading ECB speeches dataset...")

    response = requests.get(
        ECB_SPEECHES_URL,
        timeout=60,
        headers={"User-Agent": "ecb-communication-llm research project"},
    )
    response.raise_for_status()

    OUTPUT_FILE.write_bytes(response.content)

    size_mb = OUTPUT_FILE.stat().st_size / (1024 ** 2)

    print(f"Saved dataset to: {OUTPUT_FILE}")
    print(f"File size: {size_mb:.2f} MB")


if __name__ == "__main__":
    download_ecb_speeches()