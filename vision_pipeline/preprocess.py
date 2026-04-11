"""
Vision Pipeline - Preprocess
Resize, normalise, and compress images before passing to agents.
Key goal: reduce API call cost by minimising image size without
losing object detection quality.
"""

import numpy as np
from loguru import logger


# Target resolution for inference (balances accuracy vs cost)
TARGET_WIDTH = 640
TARGET_HEIGHT = 640
JPEG_QUALITY = 85  # 0-100; 85 is a good cost/quality tradeoff


def preprocess_image(image: np.ndarray, target_size: tuple = (TARGET_WIDTH, TARGET_HEIGHT)) -> np.ndarray:
    """
    Resize and normalise an image for inference.

    Args:
        image: Raw np.ndarray in BGR format
        target_size: (width, height) tuple

    Returns:
        Preprocessed np.ndarray
    """
    import cv2

    original_shape = image.shape[:2]

    # 1. Resize with aspect ratio preserved (letterbox)
    resized = letterbox(image, target_size, cv2)

    # 2. Optional: denoise for low-quality warehouse camera feeds
    # resized = cv2.fastNlMeansDenoisingColored(resized, None, 10, 10, 7, 21)

    logger.debug(f"Preprocessed: {original_shape} → {resized.shape[:2]}")
    return resized


def letterbox(image: np.ndarray, target_size: tuple, cv2) -> np.ndarray:
    """
    Resize image to target size with letterbox padding (keeps aspect ratio).
    Standard technique used by YOLO inference pipelines.
    """
    h, w = image.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target size
    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
    pad_x = (tw - new_w) // 2
    pad_y = (th - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas


def compress_to_bytes(image: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """
    Compress image to JPEG bytes for API submission.
    Reduces Cloud Vision API data transfer costs.

    Args:
        image: np.ndarray
        quality: JPEG quality (1-100)

    Returns:
        JPEG bytes
    """
    import cv2
    _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    compressed = buffer.tobytes()
    logger.debug(f"Compressed image: {len(compressed) / 1024:.1f} KB")
    return compressed


def batch_preprocess(images: list[np.ndarray], target_size: tuple = (TARGET_WIDTH, TARGET_HEIGHT)) -> list[np.ndarray]:
    """
    Preprocess a batch of images.
    Efficient for cost-controlled batch Cloud Vision API calls.
    """
    return [preprocess_image(img, target_size) for img in images]
