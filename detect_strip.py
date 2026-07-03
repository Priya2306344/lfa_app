import cv2
import numpy as np


# -------------------------
# ORDER RECTANGLE POINTS
# -------------------------
def order_points(pts):

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# -------------------------
# CROP STRIP
# -------------------------
def crop_strip(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    kernel = np.ones((5, 5), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best_rect = None
    max_area = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < 5000:
            continue

        peri = cv2.arcLength(cnt, True)

        approx = cv2.approxPolyDP(
            cnt,
            0.02 * peri,
            True
        )

        if len(approx) == 4:

            x, y, w, h = cv2.boundingRect(approx)

            aspect_ratio = w / float(h)

            if 2 < aspect_ratio < 8:

                if area > max_area:
                    max_area = area
                    best_rect = approx

    if best_rect is None:
        return img

    pts = best_rect.reshape(4, 2)

    rect = order_points(pts)

    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)

    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(
        img,
        M,
        (maxWidth, maxHeight)
    )

    return warped


# -------------------------
# DETECT LINES (Improved)
# -------------------------
def detect_lines(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    gray = cv2.equalizeHist(gray)

    # Reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        8
    )

    # Remove noise
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Vertical projection
    projection = np.sum(binary, axis=0)

    threshold = np.max(projection) * 0.18

    lines = []
    start = None

    for i in range(len(projection)):

        if projection[i] > threshold:

            if start is None:
                start = i

        else:

            if start is not None:

                end = i

                if end - start > 4:

                    center = (start + end) // 2

                    intensity = np.mean(gray[:, start:end])

                    lines.append({
                        "line_number": len(lines) + 1,
                        "position": int(center),
                        "width": int(end - start),
                        "intensity": round(255 - intensity, 2)
                    })

                start = None

    return lines


# -------------------------
# PROCESS IMAGE
# -------------------------
def process_image(file):

    file_bytes = np.frombuffer(file.read(), np.uint8)

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return {
            "error": "Unable to read image"
        }

    # Crop strip
    cropped = crop_strip(img)

    # Detect lines
    lines = detect_lines(cropped)

    # Decide status
    if len(lines) == 0:
        status = "Invalid"

    elif len(lines) == 1:
        status = "Negative"

    elif len(lines) == 2:
        status = "Positive"

    elif len(lines) == 3:
        status = "Strong Positive"

    else:
        status = "Unknown"

    return {
        "lines_detected": len(lines),
        "line_details": lines,
        "status": status
    }