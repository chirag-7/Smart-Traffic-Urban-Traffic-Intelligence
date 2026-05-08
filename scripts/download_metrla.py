# scripts/download_metrla.py
import urllib.request
import zipfile
import os

os.makedirs("data/metr-la", exist_ok=True)

print("Downloading METR-LA (~50MB)...")
urllib.request.urlretrieve(
    "https://zenodo.org/record/5724774/files/METR-LA.zip",
    "data/metr-la/METR-LA.zip"
)

print("Extracting...")
with zipfile.ZipFile("data/metr-la/METR-LA.zip", 'r') as z:
    z.extractall("data/metr-la/")

print("Done. Files:")
print(os.listdir("data/metr-la/"))