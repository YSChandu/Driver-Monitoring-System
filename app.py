# import os
# import cv2
# import numpy as np
# import face_recognition
# import dlib
# import time
# import base64
# import requests
# from datetime import datetime
# from threading import Thread
# from flask import Flask, request, jsonify
# import logging
# from imutils import face_utils
# from scipy.spatial import distance

# # Flask app
# logging.basicConfig(level=logging.INFO)
# app = Flask(__name__)

# # Constants and files
# ENCODINGS_FILE = "known_encodings.npy"
# NAMES_FILE = "known_names.npy"
# FACE_DIR = "registered_faces"
# os.makedirs(FACE_DIR, exist_ok=True)

# # Load known faces
# if os.path.exists(ENCODINGS_FILE) and os.path.exists(NAMES_FILE):
#     known_encodings = list(np.load(ENCODINGS_FILE, allow_pickle=True))
#     known_names = list(np.load(NAMES_FILE, allow_pickle=True))
# else:
#     known_encodings = []
#     known_names = []

# # Express.js backend URL (adjust as needed)
# EXPRESS_BACKEND_URL = "http://localhost:3000/api/monitoring"  # Updated URL

# # Save known data
# def save_data():
#     np.save(ENCODINGS_FILE, known_encodings)
#     np.save(NAMES_FILE, known_names)

# # Helper: send data to Express.js backend asynchronously
# def send_to_express(endpoint, data):
#     def _send():
#         try:
#             url = f"{EXPRESS_BACKEND_URL}/{endpoint}"
#             headers = {'Content-Type': 'application/json'}
            
#             # Add driver_id and trip_id if they're available
#             if 'user_name' in data:
#                 # You'll need to implement this function to get driver_id from username
#                 driver_id = get_driver_id(data['user_name'])  
#                 data['driver_id'] = driver_id
            
#             response = requests.post(url, json=data, headers=headers, timeout=5)
#             if response.status_code != 200 and response.status_code != 201:
#                 print(f"[ERROR] Express backend returned status {response.status_code}")
#                 print(f"Response: {response.text}")
#         except Exception as e:
#             print(f"[ERROR] Failed to send data to Express.js: {e}")
#     Thread(target=_send).start()

# # Face detection and encoding from a frame
# def detect_faces_and_encodings(frame):
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     face_locations = face_recognition.face_locations(rgb_frame)
#     encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#     return face_locations, encodings

# # Endpoint to register a new face with a name and image (base64 or URL)
# @app.route('/register_face', methods=['POST'])
# def register_face():
#     data = request.json
#     name = data.get('name')
#     image_b64 = data.get('image_b64')  # Expect base64 encoded image string

#     if not name or not image_b64:
#         return jsonify({"error": "Missing name or image_b64"}), 400

#     # Decode base64 image
#     import base64
#     import io
#     from PIL import Image
#     try:
#         img_data = base64.b64decode(image_b64)
#         img = Image.open(io.BytesIO(img_data))
#         frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#     except Exception as e:
#         return jsonify({"error": f"Invalid image data: {e}"}), 400

#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     face_locations = face_recognition.face_locations(rgb_frame)

#     if len(face_locations) != 1:
#         return jsonify({"error": "Exactly one face must be visible for registration"}), 400

#     face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]

#     if name in known_names:
#         return jsonify({"error": f"Name '{name}' already registered"}), 400

#     known_encodings.append(face_encoding)
#     known_names.append(name)
#     save_data()
    
#     top, right, bottom, left = face_locations[0]
#     face_image = frame[top:bottom, left:right]
#     cv2.imwrite(os.path.join(FACE_DIR, f"{name}.jpg"), face_image)

#     # Notify Express.js backend of new registration
#     send_to_express("driver/register", {
#         "name": name,
#         "face_encoding": face_encoding.tolist(),  # Convert numpy array to list
#         "registration_time": datetime.now().isoformat()
#     })

#     return jsonify({"message": f"{name} registered successfully"}), 200


# @app.route('/start_stream', methods=['POST'])
# def start_stream():
#     data = request.get_json()
#     # Debug print the received payload
#     print("Received data:", data)

#     if data is None:
#         return jsonify({'error': 'No JSON received'}), 400

#     if data.get('type') == 'wifi':
#         wifi_url = data.get('source')
#         print("Wi-Fi Camera URL received:", wifi_url)  # <-- Debug print
#         # Here you can add further logic to start streaming from this URL
#         return jsonify({'status': 'success', 'message': f'Wi-Fi URL {wifi_url} received'}), 200
#     else:
#         # Handle other camera types if needed
#         return jsonify({'status': 'success', 'message': 'Non-wifi camera selected'}), 200

# @app.route('/detect', methods=['POST'])
# def detect():
#     data = request.json
#     logging.info('Received frame from client.')
#     if not data or 'image' not in data:
#         logging.error('No image data received.')
#         return jsonify({'error': 'No image data'}), 400

#     try:
#         image_data = base64.b64decode(data['image'])
#         logging.info('Image data decoded successfully.')
#         nparr = np.frombuffer(image_data, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         if img is None:
#             logging.error('Failed to decode image.')
#             return jsonify({'error': 'Failed to decode image'}), 400
#         # Your detection logic here
#         logging.info('Detection completed.')
#         return jsonify({'result': 'detection results'})
#     except Exception as e:
#         logging.error('Error processing frame:', exc_info=True)
#         return jsonify({'error': str(e)}), 500


# # Endpoint to start monitoring on a given camera stream URL
# @app.route('/start_monitoring', methods=['POST'])
# def start_monitoring():
#     data = request.json
#     stream_url = data.get('stream_url')
#     user_name = data.get('user_name')
#     trip_id = data.get('trip_id')  # Add trip_id parameter

#     if not stream_url or not user_name:
#         return jsonify({"error": "Missing stream_url or user_name"}), 400

#     # Check if user is registered
#     if user_name not in known_names:
#         return jsonify({"error": "User not registered"}), 400

#     # Start monitoring in a separate thread
#     Thread(target=monitor_stream, args=(stream_url, user_name, trip_id)).start()

#     return jsonify({
#         "message": "Monitoring started",
#         "user_name": user_name,
#         "trip_id": trip_id
#     }), 200

# # Monitoring function
# def monitor_stream(stream_url, user_name, trip_id):
#     print(f"[INFO] Starting monitoring for {user_name} on stream {stream_url}")

#     detector = dlib.get_frontal_face_detector()
#     predictor_path = "shape_predictor_68_face_landmarks.dat"
#     if not os.path.exists(predictor_path):
#         print("[ERROR] dlib shape predictor file missing")
#         return
#     predictor = dlib.shape_predictor(predictor_path)
#     (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
#     (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#     cap = cv2.VideoCapture(stream_url)
#     if not cap.isOpened():
#         print("[ERROR] Cannot open video stream")
#         return

#     EAR_THRESHOLD = 0.25
#     CONSEC_FRAMES = 20
#     DISTRACTION_THRESHOLD = 75
#     FACE_MISSING_THRESHOLD = 15

#     DROWSINESS_ALERT_INTERVAL = 3
#     DISTRACTION_ALERT_INTERVAL = 3

#     counter = 0
#     face_missing_frames = 0

#     event_log = []

#     while True:
#         ret, frame = cap.read()
#         if not ret or frame is None:
#             print("[ERROR] Failed to grab frame")
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         rects = detector(gray, 0)

#         alerts_this_frame = []

#         if len(rects) == 0:
#             face_missing_frames += 1
#             if face_missing_frames >= FACE_MISSING_THRESHOLD:
#                 alerts_this_frame.append("[ALERT] Distraction detected! (No face visible)")
#         else:
#             face_missing_frames = 0
#             for rect in rects:
#                 shape = predictor(gray, rect)
#                 shape = face_utils.shape_to_np(shape)

#                 leftEye = shape[lStart:lEnd]
#                 rightEye = shape[rStart:rEnd]
#                 leftEAR = eye_aspect_ratio(leftEye)
#                 rightEAR = eye_aspect_ratio(rightEye)
#                 ear = (leftEAR + rightEAR) / 2.0

#                 if ear < EAR_THRESHOLD:
#                     counter += 1
#                     if counter >= CONSEC_FRAMES:
#                         alerts_this_frame.append("[ALERT] Drowsiness detected!")
#                 else:
#                     counter = 0

#                 # Head pose estimation for distraction (yaw)
#                 pitch, yaw, roll = get_head_pose(shape, frame.shape)
#                 if abs(yaw) > DISTRACTION_THRESHOLD:
#                     alerts_this_frame.append("[ALERT] Distraction detected! (Yaw angle)")

#         timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         for alert in alerts_this_frame:
#             log_entry = f"{timestamp} {alert}"
#             print(log_entry)
#             event_log.append(log_entry)

#         # Detect faces and send encodings to Express.js backend
#         face_locations, encodings = detect_faces_and_encodings(frame)
#         encodings_list = [enc.tolist() for enc in encodings]  # Convert numpy arrays to lists

#         if encodings_list:
#             payload = {
#                 "driver_id": get_driver_id(user_name),
#                 "timestamp": timestamp,
#                 "event_type": "monitoring",
#                 "event_details": {
#                     "face_encodings": encodings_list,
#                     "alerts": alerts_this_frame,
#                     "logs": event_log[-10:]
#                 }
#             }
#             send_to_express("data", payload)  # Updated endpoint

#         # Sleep briefly to reduce CPU load
#         time.sleep(0.1)

#     cap.release()
#     print(f"[INFO] Monitoring ended for {user_name}")

# # Utility functions from original code

# def eye_aspect_ratio(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     return (A + B) / (2.0 * C)

# def get_head_pose(shape, frame_size):
#     image_points = np.array([
#         shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]
#     ], dtype="double")

#     model_points = np.array([
#         (0.0, 0.0, 0.0),
#         (0.0, -330.0, -65.0),
#         (-225.0, 170.0, -135.0),
#         (225.0, 170.0, -135.0),
#         (-150.0, -150.0, -125.0),
#         (150.0, -150.0, -125.0)
#     ])

#     size = frame_size
#     focal_length = size[1]
#     center = (size[1] / 2, size[0] / 2)
#     camera_matrix = np.array([
#         [focal_length, 0, center[0]],
#         [0, focal_length, center[1]],
#         [0, 0, 1]
#     ], dtype="double")
#     dist_coeffs = np.zeros((4, 1))

#     success, rotation_vector, translation_vector = cv2.solvePnP(
#         model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

#     rmat, _ = cv2.Rodrigues(rotation_vector)
#     angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
#     return angles

# # Add this helper function
# def get_driver_id(username):
#     try:
#         response = requests.get(f"{EXPRESS_BACKEND_URL}/driver/{username}")
#         if response.status_code == 200:
#             return response.json()['driver_id']
#         return None
#     except:
#         return None

# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5000, debug=True)


from flask import Flask, render_template, request, redirect, url_for, session, Response
from driver_monitor import register, login, generate_frames  # replace with correct filename
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key'
alerts = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register_route():
    if request.method == 'POST':
        name = request.form['username'].strip()
        if not name:
            return "Name is required", 400
        register(name)
        # Pass a success message as a query parameter
        return redirect(url_for('register_route', success=1))
    # Get the success flag from query params
    success = request.args.get('success')
    return render_template('register.html', success=success)

@app.route('/login', methods=['GET', 'POST'])
def login_route():
    if request.method == 'POST':
        user = login()  # OpenCV face match
        if user:
            session['username'] = user
            return redirect(url_for('dashboard'))
        return render_template('login.html', error="Face not recognized. Try again.")
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('dashboard.html', username=session['username'])

@app.route('/video_feed')
def video_feed():
    if 'username' not in session:
        return "Unauthorized", 401
    return Response(generate_frames(alerts, session['username']),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alerts')
def get_alerts():
    return '<br>'.join(alerts[-10:])

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
