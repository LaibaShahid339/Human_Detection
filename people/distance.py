import cv2
import numpy as np

# Load the pre-trained MobileNet SSD model
net = cv2.dnn.readNetFromCaffe('C:\code projects\people\deploy.prototxt', 'C:\code projects\people\mobilenet_iter_73000.caffemodel')

# Camera calibration parameters (replace with your own values)
fx = 1000.0  # focal length in x-direction
fy = 1000.0  # focal length in y-direction
cx = 320.0   # principal point x-coordinate
cy = 240.0   # principal point y-coordinate

def calculate_distance(box1, box2):
    # Assuming you have object sizes in the real world (in meters)
    object_size_real_world = 1.0  # Replace with the actual size of your object

    # Calculate object size in pixels for both objects
    object_size_pixels_box1 = max(box1[2] - box1[0], box1[3] - box1[1])
    object_size_pixels_box2 = max(box2[2] - box2[0], box2[3] - box2[1])

    # Calculate average object size in pixels
    avg_object_size_pixels = (object_size_pixels_box1 + object_size_pixels_box2) / 2.0

    # Calculate distance using the formula
    distance = (object_size_real_world * fx) / avg_object_size_pixels

    return distance

def detect_humans():
    # Open the laptop camera
    cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera (you can change it if you have multiple cameras)

    while cap.isOpened():
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            break

        # Get the frame dimensions
        h, w = frame.shape[:2]

        # Preprocess the frame for the MobileNet SSD model
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

        # Set the input to the model and perform a forward pass
        net.setInput(blob)
        detections = net.forward()

        # Loop over the detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Check if the detection is a human (class 15 in the Coco dataset)
            if confidence > 0.2 and int(detections[0, 0, i, 1]) == 15:
                # Scale the bounding box coordinates back to the original frame size
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype(int)

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, "Human", (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Check distance between humans
                for j in range(i + 1, detections.shape[2]):
                    confidence2 = detections[0, 0, j, 2]
                    if confidence2 > 0.2 and int(detections[0, 0, j, 1]) == 15:
                        box2 = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
                        (startX2, startY2, endX2, endY2) = box2.astype(int)

                        # Calculate distance
                        distance = calculate_distance((startX, startY, endX, endY), (startX2, startY2, endX2, endY2))

                        # If the distance is less than 5 feet, issue a warning
                        if distance < 5:
                            cv2.putText(frame, f"Warning: Too Close! Distance: {distance:.2f} meters", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the result
        cv2.imshow('Human Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage
detect_humans()
