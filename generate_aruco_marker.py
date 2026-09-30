"""
Generate ArUco Marker Image for Testing
Run this script to generate an ArUco marker image that you can display on your
phone screen or print out on paper to test with your laptop webcam!
"""

import cv2
import numpy as np

MARKER_ID = 0
MARKER_SIZE_PIXELS = 400
DICT_TYPE = cv2.aruco.DICT_4X4_50

aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_TYPE)

# Generate marker image
marker_img = cv2.aruco.generateImageMarker(aruco_dict, MARKER_ID, MARKER_SIZE_PIXELS)

# Add a white border around the marker so detectors can find the black border easily
border_size = 50
bordered_img = cv2.copyMakeBorder(
    marker_img, border_size, border_size, border_size, border_size,
    cv2.BORDER_CONSTANT, value=[255, 255, 255]
)

# Add text label
cv2.putText(bordered_img, f"ArUco DICT_4X4_50 ID: {MARKER_ID}", (40, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

output_filename = f"aruco_marker_id_{MARKER_ID}.png"
cv2.imwrite(output_filename, bordered_img)
print(f"[SUCCESS] Saved test marker to: {output_filename}")
print(f"You can open this image on your phone or computer screen and hold it up to your camera!")
