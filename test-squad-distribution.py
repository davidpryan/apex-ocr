"""
Tests for RankedLoadingDetector.detect_squad_distribution.

Asserts exact counts against all four reference screens.
Run with:  python3 test-squad-distribution.py
"""

import sys
import cv2
import easyocr

from detectors import RankedLoadingDetector

CASES = [
    ("images/ui-images/ranked-loading-1.png",
     {"Diamond": 13, "Master": 4, "Apex Predator": 3}),
    ("images/ui-images/ranked-loading-2.png",
     {"Diamond": 16, "Master": 3, "Apex Predator": 1}),
    ("images/ui-images/ranked-loading-3.png",
     {"Diamond": 16, "Master": 3, "Apex Predator": 1}),
    ("images/ui-images/ranked-loading-4.png",
     {"Diamond": 20}),
]

print("Loading EasyOCR (CPU)…")
reader = easyocr.Reader(["en"], gpu=False)
det    = RankedLoadingDetector(reader)

passed = 0
failed = 0

for path, expected in CASES:
    img    = cv2.imread(path)
    result = det.detect_squad_distribution(img)
    ok     = result == expected
    status = "PASS" if ok else "FAIL"
    print(f"\n[{status}] {path}")
    print(f"  expected: {expected}")
    print(f"  got     : {result}")
    if ok:
        passed += 1
    else:
        failed += 1

print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed")
if failed:
    sys.exit(1)
