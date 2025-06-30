import cv2, time
import numpy as np
import face_recognition, dlib
from datetime import datetime
from scipy.spatial import distance
from imutils import face_utils
import mediapipe as mp
import os

FACE_DIR = "faces"
if not os.path.exists(FACE_DIR):
    os.makedirs(FACE_DIR)


def save_data():
    np.save(ENCODINGS_FILE, known_encodings)
    np.save(NAMES_FILE, known_names)

def register(name):  # now passed from Flask
    if name in known_names:
        print(f"[INFO] {name} is already registered.")
        return

    cap = cv2.VideoCapture(0)
    print("[INFO] Look at the camera to register your face...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture image")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) == 1:
            face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
            known_encodings.append(face_encoding)
            known_names.append(name)
            save_data()

            # Save face image for reference
            top, right, bottom, left = face_locations[0]
            face_image = frame[top:bottom, left:right]
            cv2.imwrite(os.path.join(FACE_DIR, f"{name}.jpg"), face_image)

            print(f"[INFO] {name} registered successfully!")
            break
        elif len(face_locations) > 1:
            print("[WARNING] Multiple faces detected. Please ensure only your face is visible.")
        else:
            print("[INFO] No face detected. Please look at the camera.")

        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Registration cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()

def login():
    if not known_encodings or not known_names:
        print("[ERROR] No registered users found. Please register first.")
        return None

    cap = cv2.VideoCapture(0)
    print("[INFO] Look at the camera to login...")

    user_name = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture image")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) == 1:
            face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.4)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            if True in matches:
                best_match_index = np.argmin(face_distances)
                user_name = known_names[best_match_index]
                print(f"[INFO] Welcome, {user_name}!")
            else:
                print("[INFO] Face not recognized. Access denied.")
            break
        elif len(face_locations) > 1:
            print("[WARNING] Multiple faces detected. Please ensure only your face is visible.")
        else:
            print("[INFO] No face detected. Please look at the camera.")

        cv2.imshow("Login Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Login cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return user_name

# === Seatbelt Detection (OpenCV line detection) ===

def detect_seatbelt(frame):
    h, w = frame.shape[:2]
    roi = frame[int(h*0.4):int(h*0.9), int(w*0.4):int(w*0.9)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

    seatbelt_lines = []
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if 0.5 < abs(slope) < 1.5:
                seatbelt_lines.append(line[0])
                cv2.line(roi, (x1,y1), (x2,y2), (0,0,255), 3)

    seatbelt_status = len(seatbelt_lines) >= 1
    return seatbelt_status


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_head_pose(shape, image_shape):
    image_points = np.array([
        shape[30],     # Nose tip
        shape[8],      # Chin
        shape[36],     # Left eye left corner
        shape[45],     # Right eye right corner
        shape[48],     # Left Mouth corner
        shape[54]      # Right mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),   # Right eye right corner
        (-150.0, -150.0, -125.0), # Left Mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ])

    focal_length = image_shape[1]
    center = (image_shape[1] / 2, image_shape[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # No lens distortion
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = euler_angles.flatten()
    return pitch, yaw, roll




# update thresholds as needed
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 8   # Lowered for more sensitivity
DISTRACTION_THRESHOLD = 35  # Lowered for more sensitivity

HAND_CONF = 0.7
SEATBELT_CHECK_INTERVAL = 60
FACE_CHECK_INTERVAL = 60

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load encodings
ENCODINGS_FILE = "known_encodings.npy"
NAMES_FILE = "known_names.npy"
if os.path.exists(ENCODINGS_FILE) and os.path.exists(NAMES_FILE):
    known_encodings = list(np.load(ENCODINGS_FILE, allow_pickle=True))
    known_names = list(np.load(NAMES_FILE, allow_pickle=True))
else:
    known_encodings, known_names = [], []

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def generate_frames(alerts, user_name="User"):
    cap = cv2.VideoCapture(0)
    last_seatbelt = time.time()
    last_face = time.time()
    last_drowsy = last_disc = last_hand = last_seatbelt_alert = last_face_alert = 0
    counter = face_missing = 0
    hands_module = mp_hands.Hands(max_num_hands=2,
                                 min_detection_confidence=HAND_CONF,
                                 min_tracking_confidence=HAND_CONF)

    # initial seatbelt check
    ret, frame = cap.read()
    if ret:
        if detect_seatbelt(frame):
            alerts.append(f"[{datetime.now():%H:%M:%S}] Seatbelt OK")
        else:
            alerts.append(f"[{datetime.now():%H:%M:%S}] ALERT: Seatbelt NOT detected")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        now = time.time()
        # periodic seatbelt
        if now - last_seatbelt >= SEATBELT_CHECK_INTERVAL:
            if not detect_seatbelt(frame) and now - last_seatbelt_alert > SEATBELT_CHECK_INTERVAL:
                alerts.append(f"[{datetime.now():%H:%M:%S}] ALERT: Seatbelt NOT detected (periodic)")
                last_seatbelt_alert = now
            last_seatbelt = now

        # hand detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands_module.process(rgb)
        if res.multi_hand_landmarks and now - last_hand > 2:
            alerts.append(f"[{datetime.now():%H:%M:%S}] ALERT: Hand detected!")
            last_hand = now
            for h in res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)

        # face, drowsiness, distraction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        if not rects:
            face_missing += 1
            if face_missing > 15 and now - last_disc > 3:
                alerts.append(f"[{datetime.now():%H:%M:%S}] ALERT: No face detected")
                last_disc = now
        else:
            face_missing = 0
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[lStart:lEnd]; rightEye = shape[rStart:rEnd]
                ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
                if ear < EAR_THRESHOLD:
                    counter += 1
                    if counter >= CONSEC_FRAMES and now - last_drowsy > 3:
                        alerts.append(f"[{datetime.now():%H:%M:%S}] ALERT: Drowsiness detected")
                        last_drowsy = now
                        counter = 0  # Reset after alert
                else:
                    counter = 0
                # distraction via head pose
                pitch, yaw, roll = get_head_pose(shape, frame.shape)
                if abs(yaw) > DISTRACTION_THRESHOLD and now - last_disc > 3:
                    alerts.append(f"[{datetime.now():%H:%M:%S}] ALERT: Distraction (Yaw)")
                    last_disc = now

        # encode frame
        ret2, buf = cv2.imencode('.jpg', frame)
        frame_bytes = buf.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()
