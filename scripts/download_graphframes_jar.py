"""
Download the GraphFrames JAR for Spark 3.5 / Scala 2.12.

The active codebase uses NetworkX in ``spark_jobs/graph_analytics.py`` instead of GraphFrames.
Keep this script if you switch graph analytics back to GraphFrames.
"""

from __future__ import annotations

import logging
import os
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

os.makedirs("spark_jobs/jars", exist_ok=True)

url = (
    "https://repos.spark-packages.org/graphframes/graphframes/"
    "0.8.3-spark3.5-s_2.12/graphframes-0.8.3-spark3.5-s_2.12.jar"
)
filepath = "spark_jobs/jars/graphframes-0.8.3-spark3.5-s_2.12.jar"

try:
    logger.info("Downloading %s", url)
    urllib.request.urlretrieve(url, filepath)
    size_mb = os.path.getsize(filepath) / (1024**2)
    logger.info("Saved %s (%.1f MB)", filepath, size_mb)
except Exception as exc:
    logger.error("Download failed: %s", exc)
    raise
