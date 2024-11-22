import cv2
import numpy as np

# Define HSV color ranges for pink and purple
pink_lower = np.array([160, 100, 100])
pink_upper = np.array([180, 255, 255])
purple_lower = np.array([110, 50, 50])
purple_upper = np.array([150, 255, 200])

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
    
    # Combine masks for pink and purple
    combined_mask = cv2.bitwise_or(pink_mask, purple_mask)
    
    # Find contours for the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour (optional: if multiple blobs might exist)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding box for the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Draw the bounding box around the combined area
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box
        
        # Display bounding box dimensions
        cv2.putText(frame, f"Width: {w}, Height: {h}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow("Combined Bounding Box", frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
