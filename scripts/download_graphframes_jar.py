import urllib.request
import os

os.makedirs("spark_jobs/jars", exist_ok=True)

url = "https://repos.spark-packages.org/graphframes/graphframes/0.8.3-spark3.5-s_2.12/graphframes-0.8.3-spark3.5-s_2.12.jar"

filepath = "spark_jobs/jars/graphframes-0.8.3-spark3.5-s_2.12.jar"

print("Downloading GraphFrames JAR (~15MB)...")
print(f"URL: {url}")

try:
    urllib.request.urlretrieve(url, filepath)
    size_mb = os.path.getsize(filepath) / (1024**2)
    print(f"✅ Downloaded: {filepath}")
    print(f"   Size: {size_mb:.1f} MB")
except Exception as e:
    print(f"❌ Error: {e}")
