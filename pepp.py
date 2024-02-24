import cv2
import mediapipe as mp

# Initialize Pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Create capture object
cap = cv2.VideoCapture('pose1.mp4')

# Get original video width and height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define new width and height for the output frame
new_width = 640  # Adjust as needed
if width != 0:  # Check for zero width to avoid division error
    new_height = int(height * (new_width / width))
else:
    new_height = 480  # Set a default height if width is zero

# Initialize video writer with the new size
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (new_width, new_height))

while cap.isOpened():
    # Read frame from capture object
    ret, frame = cap.read()

    try:
        if not ret:
            break

        # Convert the frame to RGB format
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the RGB frame to get the result
        results = pose.process(RGB)
        pose_landmarks = results.pose_landmarks
        
        if pose_landmarks:
            # Draw detected skeleton on the frame
            mp_drawing.draw_landmarks(
                frame, pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=6),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5))
            
            # Extract XYZ coordinates as integers in list format
            xyz_coordinates = [[int(lm.x * new_width), int(lm.y * new_height), int(lm.z * new_width)] for lm in pose_landmarks.landmark]
            print(xyz_coordinates)

        # Resize frame to fit the output screen
        frame = cv2.resize(frame, (new_width, new_height))
        
        # Show the final output
        cv2.imshow('Output', frame)
        out.write(frame)  # Write frame to the output video

    except Exception as e:
        print(f"Error: {e}")
        break

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()  # Release the output video writer
cv2.destroyAllWindows()
