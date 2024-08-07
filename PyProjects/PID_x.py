import cv2
import numpy as np
from simple_pid import PID # type: ignore

# PID setup
frame_width = 640  # Adjust to your camera frame width
center_x = frame_width / 2
pid = PID(0.1, 0.01, 0.005, setpoint=center_x)  # Adjusted PID gains
pid.output_limits = (-100, 100)  # Adjust as needed for your application

def apply_mask(frame):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of red color in HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks. Threshold the HSV image to get only red colors
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine the two masks
    mask = cv2.bitwise_or(mask1, mask2)

    # Bitwise-AND mask and original image
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result, mask

def find_and_draw_contours(frame, mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, return the original frame
    if not contours:
        return frame, None

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box coordinates of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Draw the bounding box on the frame if width is 30 or greater
    if w >= 30:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Calculate the center x-coordinate of the bounding box
    center_x_box = x + w / 2

    return frame, center_x_box

def livestream_from_camera(camera_index=0):
    # Open a connection to the camera
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image")
            break

        # Apply mask to the frame
        masked_frame, mask = apply_mask(frame)

        # Find and draw contours on the masked frame
        frame_with_contours, center_x_box = find_and_draw_contours(masked_frame, mask)

        if center_x_box is not None:
            # Compute the control output using PID
            control = pid(center_x_box)
            # Output the center x and control values to the terminal
            print(f'Center X: {center_x_box:.2f}, Control: {control:.2f}, Error: {center_x_box - center_x:.2f}')

        # Display the resulting frame and mask
        cv2.imshow('Live Stream with Contours', frame_with_contours)
        cv2.imshow('Mask', mask)

        # Press 'q' to quit the livestream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    livestream_from_camera()
