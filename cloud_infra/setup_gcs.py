"""
Cloud Infrastructure - Google Cloud Storage
Sets up GCS bucket, enables required APIs, and handles upload/download.
Run this once before starting the pipeline: python -m cloud_infra.setup_gcs
"""

import os
import json
from pathlib import Path
from loguru import logger


PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-project-id")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "warehouse-visual-intelligence")
REGION = os.getenv("GCP_REGION", "us-central1")


def create_bucket(bucket_name: str = BUCKET_NAME, region: str = REGION) -> bool:
    """Create a GCS bucket if it doesn't exist."""
    try:
        from google.cloud import storage
        client = storage.Client(project=PROJECT_ID)

        if client.bucket(bucket_name).exists():
            logger.info(f"Bucket already exists: gs://{bucket_name}")
            return True

        bucket = client.create_bucket(bucket_name, location=region)
        logger.success(f"Bucket created: gs://{bucket.name} in {region}")
        return True

    except Exception as e:
        logger.error(f"Failed to create bucket: {e}")
        return False


def upload_image(local_path: Path, gcs_folder: str = "input/") -> str | None:
    """
    Upload a local image to GCS.

    Returns:
        GCS URI (gs://bucket/path) or None on failure
    """
    try:
        from google.cloud import storage
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)
        blob_name = gcs_folder + local_path.name
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        uri = f"gs://{BUCKET_NAME}/{blob_name}"
        logger.success(f"Uploaded: {local_path.name} → {uri}")
        return uri
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return None


def upload_report(report_path: Path, gcs_folder: str = "reports/") -> str | None:
    """Upload a JSON report to GCS."""
    return upload_image(report_path, gcs_folder=gcs_folder)


def download_images(gcs_folder: str = "input/", local_dir: Path = Path("data/downloaded/")) -> list[Path]:
    """Download all images from a GCS folder to a local directory."""
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    try:
        from google.cloud import storage
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)
        blobs = list(bucket.list_blobs(prefix=gcs_folder))
        for blob in blobs:
            local_path = local_dir / Path(blob.name).name
            blob.download_to_filename(str(local_path))
            downloaded.append(local_path)
            logger.debug(f"Downloaded: {blob.name}")
        logger.info(f"Downloaded {len(downloaded)} file(s) to {local_dir}")
    except Exception as e:
        logger.error(f"Download failed: {e}")
    return downloaded


if __name__ == "__main__":
    logger.info("Setting up Google Cloud Storage...")
    logger.info(f"Project: {PROJECT_ID}")
    logger.info(f"Bucket:  gs://{BUCKET_NAME}")
    create_bucket()
    logger.info("Done. Set GCP_PROJECT_ID and GCS_BUCKET_NAME in your .env file.")
