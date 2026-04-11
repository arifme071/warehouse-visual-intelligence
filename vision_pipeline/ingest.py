"""
Vision Pipeline - Ingest
Load images from local path or Google Cloud Storage bucket.
"""

import os
import numpy as np
from pathlib import Path
from loguru import logger


SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_images(source: Path) -> list[np.ndarray]:
    """
    Load images from a file path or directory.

    Args:
        source: Path to a single image or a directory of images

    Returns:
        List of images as numpy arrays (BGR, uint8)
    """
    import cv2

    images = []

    if source.is_file():
        img = _load_single(source, cv2)
        if img is not None:
            images.append(img)

    elif source.is_dir():
        paths = sorted(
            [p for p in source.iterdir() if p.suffix.lower() in SUPPORTED_FORMATS]
        )
        logger.info(f"Found {len(paths)} image(s) in {source}")
        for p in paths:
            img = _load_single(p, cv2)
            if img is not None:
                images.append(img)
    else:
        logger.error(f"Source not found: {source}")

    return images


def load_from_gcs(bucket_name: str, prefix: str = "") -> list[np.ndarray]:
    """
    Load images directly from a Google Cloud Storage bucket.

    Args:
        bucket_name: GCS bucket name (without gs://)
        prefix: Optional folder prefix inside the bucket

    Returns:
        List of images as numpy arrays
    """
    import cv2
    import numpy as np
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    images = []
    for blob in blobs:
        ext = Path(blob.name).suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            continue
        data = blob.download_as_bytes()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
            logger.debug(f"Loaded from GCS: {blob.name}")

    logger.info(f"Loaded {len(images)} image(s) from gs://{bucket_name}/{prefix}")
    return images


def _load_single(path: Path, cv2) -> np.ndarray | None:
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        logger.warning(f"Skipping unsupported file: {path.name}")
        return None
    img = cv2.imread(str(path))
    if img is None:
        logger.warning(f"Failed to load: {path}")
        return None
    logger.debug(f"Loaded: {path.name} ({img.shape[1]}x{img.shape[0]})")
    return img
