import cv2
import numpy as np

# Known width of the object in real-world (in cm)
KNOWN_WIDTH = 10.2  # Width of mouse in cm

# Focal length of the camera (in the same units as KNOWN_WIDTH)
FOV = 9.3  # Field of View in cm
WORKING_DISTANCE = 7.7  # Working distance in cm
SENSOR_SIZE = 0.635  # Sensor size in cm

# Calculate the focal length based on the provided parameters
FOCAL_LENGTH = ((SENSOR_SIZE * WORKING_DISTANCE) / FOV)*1000
#FOCAL_LENGTH = 500
print(f"Calculated Focal Length: {FOCAL_LENGTH:.2f} cm")

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
        return frame

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box coordinates of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Print the perceived width for debugging
    print(f"Perceived Width (pixels): {w}")

    # Draw the bounding box on the frame if width is 30 or greater
    if w >= 30:
        # Calculate the distance to the object
        distance = (KNOWN_WIDTH * FOCAL_LENGTH) / w

        # Print the calculated distance for debugging
        print(f"Calculated Distance: {distance:.2f} cm")

        # Draw the bounding box and distance on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'Distance: {distance:.2f} cm', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

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
        frame_with_contours = find_and_draw_contours(masked_frame, mask)

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
