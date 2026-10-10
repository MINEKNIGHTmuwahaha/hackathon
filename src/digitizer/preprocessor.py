"""
src/digitizer/preprocessor.py
-----------------------------
Gentle photo cleanup for ECG images.

Deliberately conservative — we do NOT want to destroy the trace.
Only fixes: perspective, contrast, mild illumination, mild denoise.
"""

from pathlib import Path
import numpy as np
import cv2

from ..logger import get_logger


log = get_logger("sentinel.preprocess_photo")


def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def detect_paper_corners(img: np.ndarray) -> np.ndarray | None:
    """Find 4 corners of the ECG paper. Returns None if uncertain."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / (h * w)
    # Only accept if paper fills >50% of the frame
    if area_ratio < 0.5:
        return None
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    if len(approx) != 4:
        return None
    return approx.reshape(4, 2).astype(np.float32)


def deskew(img: np.ndarray) -> np.ndarray:
    """Warp the paper rectangle to flat. Skips if detection fails."""
    corners = detect_paper_corners(img)
    if corners is None:
        log.info("deskew: paper not detected, skipping")
        return img

    rect = order_corners(corners)
    (tl, tr, br, bl) = rect
    maxW = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    maxH = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    if maxW < 200 or maxH < 100:
        return img

    dst = np.array([[0, 0], [maxW - 1, 0],
                    [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (maxW, maxH))


def mild_illumination_fix(img: np.ndarray) -> np.ndarray:
    """
    Very gentle shadow removal. Blends 30% of the corrected version
    with 70% original — avoids destroying the image.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Big Gaussian blur approximates background illumination
    bg = cv2.GaussianBlur(l, (0, 0), sigmaX=40, sigmaY=40)
    corrected = cv2.addWeighted(l, 1.0, bg, -0.5, 128)

    # Blend only 30% of the correction
    blended = cv2.addWeighted(l, 0.7, corrected, 0.3, 0)

    lab = cv2.merge([blended, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def mild_contrast(img: np.ndarray, clip: float = 1.5) -> np.ndarray:
    """Very mild CLAHE — clip=1.5 is safe (was 2.5 = too aggressive)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def clean_photo(image_path: str,
                out_path: str | None = None,
                do_deskew: bool = True,
                do_illumination: bool = True,
                do_contrast: bool = True) -> dict:
    """
    Gentle cleanup pipeline. No denoise, no binarization, no gamma boost —
    those destroy ECG traces in practice.
    """
    p = Path(image_path)
    if not p.exists():
        return {"ok": False, "error": f"File not found: {p}",
                "output_path": None, "steps_applied": []}

    img = cv2.imread(str(p))
    if img is None:
        return {"ok": False, "error": "Could not read image",
                "output_path": None, "steps_applied": []}

    steps = []

    if do_deskew:
        before_shape = img.shape
        img = deskew(img)
        if img.shape != before_shape:
            steps.append("deskew")

    if do_illumination:
        img = mild_illumination_fix(img)
        steps.append("illumination")

    if do_contrast:
        img = mild_contrast(img, clip=1.5)
        steps.append("contrast")

    if out_path is None:
        out_path = str(p.with_stem(p.stem + "_clean"))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)

    return {
        "ok": True,
        "error": None,
        "output_path": str(out),
        "steps_applied": steps,
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/samples/test_pvc_phone.jpg"
    result = clean_photo(path)
    if result["ok"]:
        print(f"✓ cleaned: {result['output_path']}")
        print(f"  steps: {', '.join(result['steps_applied'])}")
    else:
        print(f"✗ {result['error']}")