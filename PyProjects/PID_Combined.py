import cv2
import numpy as np
from simple_pid import PID  # type: ignore

# PID setup for x, y, and depth positions
frame_width = 640  # Adjust to your camera frame width
frame_height = 480  # Adjust to your camera frame height
center_x = frame_width / 2
center_y = frame_height / 2
setpoint_depth = 254  # Example setpoint for depth in millimeters

pid_x = PID(0.1, 0.01, 0.005, setpoint=center_x)  # PID for horizontal position
pid_y = PID(0.1, 0.01, 0.005, setpoint=center_y)  # PID for vertical position
pid_depth = PID(1.0, 0.1, 0.05, setpoint=setpoint_depth)  # PID for depth position

pid_x.output_limits = (-100, 100)  # Adjust as needed for your application
pid_y.output_limits = (-100, 100)  # Adjust as needed for your application
pid_depth.output_limits = (-100, 100)  # Adjust as needed for your application

def apply_mask(frame):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Define range of neon yellow color in HSV
    lower_yellow = np.array([20, 100, 100])  # Lower bound for neon yellow
    upper_yellow = np.array([30, 255, 255])  # Upper bound for neon yellow

    # Create masks. Threshold the HSV image to get only neon yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Bitwise-AND mask and original image
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result, mask

def find_and_draw_contours(frame, mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, return the original frame
    if not contours:
        return frame, None, None, None

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box coordinates of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Calculate the center coordinates of the bounding box
    center_x_box = x + w // 2
    center_y_box = y + h // 2

    # Draw the center point of the bounding box
    center_point = (int(center_x_box), int(center_y_box))
    cv2.circle(frame, center_point, 5, (255, 0, 0), -1)  # Blue circle for center point

    # Estimate depth (for demonstration, replace with actual depth calculation if available)
    depth = w  # Example: using width as a proxy for depth

    return frame, center_x_box, center_y_box, depth

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
        frame_with_contours, center_x_box, center_y_box, depth = find_and_draw_contours(masked_frame, mask)

        if center_x_box is not None and center_y_box is not None:
            # Compute the control outputs using PID
            control_x = pid_x(center_x_box)
            control_y = pid_y(center_y_box)
            control_depth = pid_depth(depth)

            # Output the center x, center y, depth, and control values to the terminal
            print(f'Center X: {center_x_box:.2f}\tControl X: {control_x:.2f}')
            print(f'Center Y: {center_y_box:.2f}\tControl Y: {control_y:.2f}')
            print(f'Depth: {depth:.2f}\t\tControl Depth: {control_depth:.2f}')

        # Display the resulting frame and mask
        cv2.imshow('Live Stream with Bounding Box and Center Point', frame_with_contours)
        cv2.imshow('Mask', mask)

        # Press 'q' to quit the livestream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    livestream_from_camera()
