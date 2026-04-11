"""
Warehouse Visual Intelligence System
Entry point - runs the full multi-agent pipeline on a given image or folder.
"""

import argparse
from pathlib import Path
from loguru import logger
from vision_pipeline.ingest import load_images
from vision_pipeline.preprocess import preprocess_image
from agents.orchestrator import Orchestrator


def parse_args():
    parser = argparse.ArgumentParser(description="Warehouse Visual Intelligence System")
    parser.add_argument("--input", type=str, required=True, help="Path to image or folder")
    parser.add_argument("--output", type=str, default="output/", help="Output directory")
    parser.add_argument("--cloud", action="store_true", help="Upload results to GCS")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting pipeline on: {input_path}")

    # 1. Load images
    images = load_images(input_path)
    logger.info(f"Loaded {len(images)} image(s)")

    # 2. Preprocess
    processed = [preprocess_image(img) for img in images]
    logger.info("Preprocessing complete")

    # 3. Run multi-agent pipeline
    orchestrator = Orchestrator()
    report = orchestrator.run(processed)

    # 4. Save report
    report_path = output_path / "report.json"
    report.save(report_path)
    logger.success(f"Report saved to: {report_path}")

    if args.cloud:
        from cloud_infra.setup_gcs import upload_report
        upload_report(report_path)
        logger.success("Report uploaded to GCS")


if __name__ == "__main__":
    main()
