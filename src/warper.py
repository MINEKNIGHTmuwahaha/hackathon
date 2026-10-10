"""Simulate a realistic phone-taken ECG photo from a clean render."""
import cv2
import numpy as np
from pathlib import Path


def simulate_phone_photo(input_path: str, output_path: str):
    img = cv2.imread(input_path)
    if img is None:
        print(f"[skip] {input_path} — not found")
        return False

    h, w = img.shape[:2]

    # 1. Mild perspective warp (phone at slight angle)
    offset = int(w * 0.012)
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst = np.float32([
        [offset, offset],
        [w - offset, 0],
        [0, h - offset],
        [w, h],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, M, (w, h),
                              borderMode=cv2.BORDER_REPLICATE)

    # 2. Light blur (out of focus)
    img = cv2.GaussianBlur(img, (3, 3), 0.6)

    # 3. Subtle brightness + contrast
    alpha = 1.05
    beta = 10
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # 4. GENTLE shadow (was too strong before)
    shadow = np.zeros((h, w), dtype=np.float32)
    # Small offset shadow instead of huge circle
    cx, cy = int(w * 0.7), int(h * 0.3)
    cv2.circle(shadow, (cx, cy), int(h * 0.9), 1.0, -1)
    shadow = cv2.GaussianBlur(shadow, (201, 201), 60)
    shadow = (shadow * 35).astype(np.uint8)  # much gentler
    for c in range(3):
        img[:, :, c] = cv2.subtract(img[:, :, c], shadow)

    # 5. Tiny rotation (hand tilt)
    angle = np.random.uniform(-1.5, 1.5)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h),
                         borderMode=cv2.BORDER_REPLICATE)

    # 6. Light JPEG compression artifact
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, enc = cv2.imencode(".jpg", img, encode_param)
    img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    cv2.imwrite(output_path, img)
    print(f"saved {output_path}")
    return True


if __name__ == "__main__":
    names = ["test_pvc", "test_normal", "test_bradycardia", "test_supraventricular"]
    for name in names:
        simulate_phone_photo(
            f"data/samples/{name}.jpg",
            f"data/samples/{name}_phone.jpg",
        )