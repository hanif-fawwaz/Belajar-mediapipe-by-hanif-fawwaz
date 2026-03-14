import cv2
import numpy as np

# Open the default webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # If frame is not read correctly, break the loop
    if not ret:
        break

    # Convert the frame from BGR to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for a specific color (e.g., blue)
    # These values can be adjusted based on the specific lighting conditions and desired color range
    lower_blue = np.array([94, 80, 2])
    upper_blue = np.array([120, 255, 255])

    # Create a mask to isolate the specified color
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    # Apply the mask to the original frame using bitwise AND operation
    # This will show only the pixels within the specified color range
    blue_detection = cv2.bitwise_and(frame, frame, mask=blue_mask)

    # Display the original frame, the mask, and the detected color
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Blue Mask', blue_mask)
    cv2.imshow('Blue Detection', blue_detection)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()