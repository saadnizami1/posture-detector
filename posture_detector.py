import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Attempt to open up to 4 cameras
cams = []
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} opened successfully.")
        cams.append(cap)
    else:
        cap.release()

if not cams:
    print("No camera found.")
    exit()

# Frame size for each camera feed in composite window
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

while True:
    angles = []
    frames = []

    for cap in cams:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            frames.append(frame)
            continue

        original = frame.copy()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w, _ = image.shape
        angle_text = "No Person"
        color = (0, 0, 255)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get key points (left shoulder, hip, knee)
            try:
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                # Convert to pixel coordinates
                shoulder = tuple(np.multiply(shoulder, [w, h]).astype(int))
                hip = tuple(np.multiply(hip, [w, h]).astype(int))
                knee = tuple(np.multiply(knee, [w, h]).astype(int))

                angle = calculate_angle(shoulder, hip, knee)
                angles.append(angle)

                angle_text = f"Angle: {int(angle)}"
                color = (0, 255, 0) if 145 <= angle <= 170 else (0, 0, 255)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            except Exception as e:
                angle_text = "Error"
                color = (0, 0, 255)

        # Draw per-camera info
        cv2.putText(image, angle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        image = cv2.resize(image, (FRAME_WIDTH, FRAME_HEIGHT))
        frames.append(image)

    # Combine all frames into a grid (1xN or 2x2)
    if len(frames) == 1:
        combined = frames[0]
    elif len(frames) == 2:
        combined = np.hstack((frames[0], frames[1]))
    elif len(frames) == 3:
        row1 = np.hstack((frames[0], frames[1]))
        row2 = np.hstack((frames[2], np.zeros_like(frames[2])))
        combined = np.vstack((row1, row2))
    elif len(frames) == 4:
        row1 = np.hstack((frames[0], frames[1]))
        row2 = np.hstack((frames[2], frames[3]))
        combined = np.vstack((row1, row2))

    # Final posture decision
    if angles:
        avg_angle = sum(angles) / len(angles)
        posture_msg = "Good posture" if 145 <= avg_angle <= 170 else "Fix your posture!"
        color = (0, 255, 0) if posture_msg == "Good posture" else (0, 0, 255)

        cv2.putText(combined, f"Avg Angle: {int(avg_angle)}", (10, combined.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(combined, posture_msg, (10, combined.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    cv2.imshow("Posture Detector - Multi Camera", combined)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

# Cleanup
for cap in cams:
    cap.release()
cv2.destroyAllWindows()