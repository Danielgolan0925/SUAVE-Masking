import cv2
import numpy as np
from simple_pid import PID

# Define HSV color ranges for pink and purple
pink_lower = np.array([160, 100, 100])
pink_upper = np.array([180, 255, 255])
purple_lower = np.array([110, 50, 50])
purple_upper = np.array([150, 255, 200])

# Initialize the PID controller
pid = PID(setpoint=0)  # Target error is 0 (equal widths)
pid.Kp = 0.1  # Proportional gain
pid.Ki = 0.01  # Integral gain
pid.Kd = 0.05  # Derivative gain
pid.output_limits = (-100, 100)  # Adjust output range for your use case

# Proximity threshold (horizontal distance in pixels)
proximity_threshold = 20

# Initialize video stream
cap = cv2.VideoCapture(0)  # Replace with your video source if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create masks for pink and purple
    pink_mask = cv2.inRange(hsv, pink_lower, pink_upper)
    purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)
    
    pink_width = 0
    purple_width = 0
    
    # Find contours for pink
    pink_contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if pink_contours:
        pink_contour = max(pink_contours, key=cv2.contourArea)  # Largest pink contour
        px, py, pw, ph = cv2.boundingRect(pink_contour)
        pink_width = pw  # Width of pink bounding box

    # Find contours for purple
    purple_contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if purple_contours:
        purple_contour = max(purple_contours, key=cv2.contourArea)  # Largest purple contour
        prx, pry, prw, prh = cv2.boundingRect(purple_contour)
        purple_width = prw  # Width of purple bounding box

    # Check proximity and vertical alignment
    if pink_width > 0 and purple_width > 0:
        # Calculate horizontal distance between pink and purple boxes
        horizontal_distance = abs((px + pw) - prx)
        
        # Check if the bounding boxes are close enough horizontally
        # vertically_aligned = not (py > pry + prh or pry > py + ph)  # Ensure vertical overlap
        #if horizontal_distance <= proximity_threshold and vertically_aligned:
        if True:
            # Draw bounding boxes if adjacent
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 255, 0), 2)  # Green box for pink
            cv2.putText(frame, f"Pink Width: {pw}", (px, py + ph + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.rectangle(frame, (prx, pry), (prx + prw, pry + prh), (255, 0, 0), 2)  # Blue box for purple
            cv2.putText(frame, f"Purple Width: {prw}", (prx, pry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Calculate error and control signal
            error = pink_width - purple_width
            control_signal = pid(error)
            print(f"Error: {error}, Control Signal: {control_signal}")
            
            # Overlay control signal on the frame
            cv2.putText(frame, f"Control Signal: {control_signal:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display the frame
    cv2.imshow("Bounding Box Tracking", frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
