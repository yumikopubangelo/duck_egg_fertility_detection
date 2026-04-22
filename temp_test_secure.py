#!/usr/bin/env python3
"""Test secure_filename behavior against path traversal attacks."""
import sys
import os
sys.path.insert(0, r'C:\Users\vanguard\OneDrive\Documents\duck_egg_fertility_detection')
from werkzeug.utils import secure_filename

test_cases = [
    ("../../../etc/passwd", "etc_passwd"),
    ("..\\..\\..\\windows\\system32", "windows_system32"),
    ("shell.php.jpg", "shell.php.jpg"),
    ("image.jpg\x00.png", "image.jpg.png"),
    ("/absolute/path/file.jpg", "absolute_path_file.jpg"),
    ("normal_image.jpg", "normal_image.jpg"),
    ("..", ""),
    ("./relative/path", "relative_path"),
]

print("Testing secure_filename():")
print("=" * 60)
for test_input, expected in test_cases:
    result = secure_filename(test_input)
    match = "PASS" if result == expected else "FAIL"
    print(f"[{match}] Input:    {repr(test_input)}")
    print(f"       Expected: {repr(expected)}")
    print(f"       Got:      {repr(result)}")
    print()
