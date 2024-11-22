import cv2
import numpy as np
from simple_pid import PID

# Frame dimensions (modify as per your camera settings)
frame_width = 640
frame_height = 480
center_x = frame_width / 2
center_y = frame_height / 2
setpoint_depth = 254  # Example depth setpoint in millimeters

# HSV color ranges for pink and purple
pink_lower = np.array([160, 100, 100])
pink_upper = np.array([180, 255, 255])
purple_lower = np.array([110, 50, 50])
purple_upper = np.array([150, 255, 200])

# PID Controllers for X, Y, Depth, and Yaw
pid_x = PID(0.1, 0.01, 0.005, setpoint=center_x)
pid_y = PID(0.1, 0.01, 0.005, setpoint=center_y)
pid_depth = PID(1.0, 0.1, 0.05, setpoint=setpoint_depth)
pid_yaw = PID(0.1, 0.01, 0.005, setpoint=0)  # Yaw setpoint is 0 (equal widths)

# Output limits
pid_x.output_limits = (-100, 100)
pid_y.output_limits = (-100, 100)
pid_depth.output_limits = (-100, 100)
pid_yaw.output_limits = (-100, 100)

# Initialize video stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for pink and purple
    pink_mask = cv2.inRange(hsv, pink_lower, pink_upper)
    purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)

    # Combine masks
    combined_mask = cv2.bitwise_or(pink_mask, purple_mask)

    # Find contours for the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Get bounding box for the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        center_x_box = x + w // 2
        center_y_box = y + h // 2

        # Calculate depth using bounding box width (example proxy for depth)
        depth = w

        # Draw bounding box and center
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (center_x_box, center_y_box), 5, (255, 0, 0), -1)
        
        # Separate bounding boxes for pink and purple to calculate yaw error
        pink_contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        purple_contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pink_width = 0
        purple_width = 0

        if pink_contours:
            pink_contour = max(pink_contours, key=cv2.contourArea)
            px, py, pw, ph = cv2.boundingRect(pink_contour)
            pink_width = pw
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 255, 255), 2)  # Yellow for pink box

        if purple_contours:
            purple_contour = max(purple_contours, key=cv2.contourArea)
            prx, pry, prw, prh = cv2.boundingRect(purple_contour)
            purple_width = prw
            cv2.rectangle(frame, (prx, pry), (prx + prw, pry + prh), (255, 0, 255), 2)  # Magenta for purple box

        # Calculate yaw error (difference between pink and purple widths)
        yaw_error = pink_width - purple_width
        yaw_control = pid_yaw(yaw_error)

        # Compute PID control outputs for X, Y, and Depth
        control_x = pid_x(center_x_box)
        control_y = pid_y(center_y_box)
        control_depth = pid_depth(depth)

        # Output control values to the terminal
        print(f'Center X: {center_x_box:.2f}\tControl X: {control_x:.2f}')
        print(f'Center Y: {center_y_box:.2f}\tControl Y: {control_y:.2f}')
        print(f'Depth: {depth:.2f}\t\tControl Depth: {control_depth:.2f}')
        print(f'Yaw Error: {yaw_error:.2f}\tControl Yaw: {yaw_control:.2f}')

        # Overlay control values on the frame
        cv2.putText(frame, f"Control X: {control_x:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Control Y: {control_y:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Control Depth: {control_depth:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Control Yaw: {yaw_control:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame and the combined mask
    cv2.imshow("Bounding Box Tracking", frame)
    cv2.imshow("Combined Mask", combined_mask)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
