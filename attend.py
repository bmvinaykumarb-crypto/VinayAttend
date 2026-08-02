from streamlit.runtime import scriptrunner
from streamlit.runtime import scriptrunner
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from streamlit_js_eval import get_geolocation, streamlit_js_eval
from geopy.distance import geodesic

COLLEGE_LOCATION = (15.273742673769599, 76.37739703526368)
ALLOWED_RADIUS_METERS = 300000

def is_within_range(lat, lon):
    distance = geodesic(COLLEGE_LOCATION, (lat, lon)).meters
    return distance <= ALLOWED_RADIUS_METERS, distance

import pandas as pd
import hashlib
import time
from datetime import datetime
import os
from pathlib import Path
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import json
import streamlit as st
import urllib.request
import urllib.error

# --- MediaPipe for face detection, landmarks (liveness) ---
try:
    import mediapipe as mp
    FACE_RECOGNITION_AVAILABLE = True
except (ImportError, AttributeError):
    FACE_RECOGNITION_AVAILABLE = False

# --- OpenCV SFace for face recognition/embedding ---
SFACE_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
SFACE_MODEL_PATH = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")
YUNET_MODEL_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
FACE_LANDMARKER_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")


def ensure_models_downloaded():
    """Download SFace, YuNet, and MediaPipe FaceLandmarker models if not already cached."""
    import ssl
    os.makedirs(MODEL_DIR, exist_ok=True)
    models = [
        (SFACE_MODEL_URL, SFACE_MODEL_PATH),
        (YUNET_MODEL_URL, YUNET_MODEL_PATH),
        (FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH),
    ]
    for url, path in models:
        if not os.path.exists(path):
            try:
                urllib.request.urlretrieve(url, path)
            except ssl.SSLError:
                # Fallback: bypass SSL verification (safe for known model URLs)
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                try:
                    with urllib.request.urlopen(url, context=ctx) as response:
                        with open(path, 'wb') as f:
                            f.write(response.read())
                except Exception as e2:
                    st.warning(f"Failed to download model from {url}: {e2}")
                    return False
            except urllib.error.URLError:
                # Also handle URLError wrapping SSL errors
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                try:
                    with urllib.request.urlopen(url, context=ctx) as response:
                        with open(path, 'wb') as f:
                            f.write(response.read())
                except Exception as e2:
                    st.warning(f"Failed to download model from {url}: {e2}")
                    return False
            except Exception as e:
                st.warning(f"Failed to download model from {url}: {e}")
                return False
    return True


def get_face_detector(image_width, image_height):
    """Get OpenCV YuNet face detector."""
    detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL_PATH, "", (image_width, image_height),
        score_threshold=0.6, nms_threshold=0.3, top_k=5000
    )
    return detector


def get_face_recognizer():
    """Get OpenCV SFace face recognizer."""
    return cv2.FaceRecognizerSF.create(SFACE_MODEL_PATH, "")


def extract_face_embedding(image_np):
    """Extract 128-d face embedding from an image using YuNet + SFace.
    Returns (embedding_array, error_string). On success error_string is None."""
    if not ensure_models_downloaded():
        return None, "Face recognition models are not available."

    h, w = image_np.shape[:2]
    detector = get_face_detector(w, h)
    recognizer = get_face_recognizer()

    _, faces = detector.detect(image_np)
    if faces is None or len(faces) == 0:
        return None, "No face detected in the image. Please make sure your face is clearly visible."
    if len(faces) > 1:
        return None, "Multiple faces detected. Please make sure only one person is in the frame."

    aligned_face = recognizer.alignCrop(image_np, faces[0])
    embedding = recognizer.feature(aligned_face)
    return embedding.flatten(), None


def compare_face_embeddings(emb1, emb2, threshold=0.363):
    """Compare two face embeddings using cosine similarity.
    Returns (is_match, similarity_score)."""
    recognizer = get_face_recognizer()
    score = recognizer.match(
        emb1.reshape(1, -1).astype(np.float32),
        emb2.reshape(1, -1).astype(np.float32),
        cv2.FaceRecognizerSF_FR_COSINE
    )
    return score >= threshold, score


# --- MediaPipe landmark indices for EAR (Eye Aspect Ratio) blink detection ---
# Right eye landmarks (6 points matching the standard EAR formula)
MP_RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Left eye landmarks
MP_LEFT_EYE = [362, 385, 387, 263, 373, 380]
# Lip landmarks for MAR (Mouth Aspect Ratio)
MP_TOP_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
MP_BOTTOM_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
# Chin/jaw outline for yaw estimation
MP_CHIN = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
# Nose tip for yaw
MP_NOSE_TIP = [1, 2, 98, 327, 4]


st.set_page_config(page_title="Smart Lab Attendance", layout="wide", page_icon="📝")

def load_static_file(filename: str) -> str:
    path = Path(__file__).parent / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""

st.subheader("📍 Location Verification")

if 'location_verified' not in st.session_state:
    st.session_state.location_verified = False
    st.session_state.location_distance = None

if not st.session_state.location_verified:
    js_code = """
    new Promise((resolve, reject) => {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    resolve({
                        coords: {
                            latitude: position.coords.latitude,
                            longitude: position.coords.longitude,
                        }
                    });
                },
                (error) => {
                    resolve({
                        error: {
                            code: error.code,
                            message: error.message,
                        }
                    });
                },
                {
                    enableHighAccuracy: true,
                    timeout: 20000,
                    maximumAge: 70000
                }
            );
        } else {
            resolve({ error: { message: 'Browser does not support geolocation!' } });
        }
    })
    """
    location = streamlit_js_eval(js_expressions=js_code, key='get_loc_fast')

    if location is None:
        st.warning("⏳ Fetching your location... Please wait. (This can take up to 25 seconds on some computers to get a GPS lock)")
        st.info("💡 **Not seeing a prompt?**\n1. Click the **lock icon (🔒/tune)** in your browser's address bar (next to localhost:8501).\n2. Ensure **Location** is turned on or set to 'Allow'.\n3. If you are on macOS, ensure Chrome has Location permissions in System Settings -> Privacy & Security.\n4. Refresh the page.")
        if st.button("🔄 Retry Location Check"):
            st.rerun()
        st.stop()
    elif 'error' in location:
        st.error(f"❌ Location error: {location['error']['message']}")
        if st.button("🔄 Retry Location Check"):
            st.rerun()
        st.stop()
    else:
        lat = location['coords']['latitude']
        lon = location['coords']['longitude']

        within_range, distance = is_within_range(lat, lon)

        if not within_range:
            st.error(f"❌ You are {distance:.0f}m  u are fare away from VNC BCA(block) college. Attendance can only be marked within this area {ALLOWED_RADIUS_METERS}m.")
            st.stop()
        else:
            st.session_state.location_verified = True
            st.session_state.location_distance = distance
            st.success(f"✅ Location verified ({distance:.0f}m from college)")
else:
    st.success(f"✅ Location verified ({st.session_state.location_distance:.0f}m from college)")

css = load_static_file("style.css")
if css:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Initialize session state for user login & redirects
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = True  # Direct access for students
if 'user_role' not in st.session_state:
    st.session_state.user_role = "student"  # Default to student
if 'username' not in st.session_state:
    st.session_state.username = "Student"
if 'redirecting' not in st.session_state:
    st.session_state.redirecting = False
if 'target_role' not in st.session_state:
    st.session_state.target_role = None
if 'auto_scan_active' not in st.session_state:
    st.session_state.auto_scan_active = False
if 'show_faculty_login' not in st.session_state:
    st.session_state.show_faculty_login = False
if 'show_admin_login' not in st.session_state:
    st.session_state.show_admin_login = False

import sqlite3

DB_FILE = "attendance.db"
CSV_FILE = "lab_attendance.csv"
STUDENT_REGISTRY_FILE = "student_registry.csv"
FACULTIES_REGISTRY_FILE = "faculties_registry.csv"
REGISTERED_FACES_DIR = "registered_faces"

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize SQLite database tables and auto-migrate existing CSV data."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Attendance Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_number TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                lab TEXT NOT NULL
            )
        """)
        
        # 2. Students Registry Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                roll_number TEXT PRIMARY KEY,
                registration_date TEXT,
                face_encoding TEXT,
                face_path TEXT
            )
        """)
        
        # 3. Faculties Registry Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS faculties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                department TEXT NOT NULL,
                year TEXT,
                semester TEXT
            )
        """)
        conn.commit()

        # Migrate attendance from CSV if DB table is empty
        cursor.execute("SELECT COUNT(*) FROM attendance")
        if cursor.fetchone()[0] == 0 and os.path.exists(CSV_FILE):
            try:
                csv_df = pd.read_csv(CSV_FILE)
                for _, row in csv_df.iterrows():
                    r_num = str(row.get("Roll Number", "")).strip()
                    r_date = str(row.get("Date", "")).strip()
                    r_time = str(row.get("Time", "")).strip()
                    r_lab = str(row.get("Lab", "")).strip()
                    if r_num and r_date and r_lab:
                        cursor.execute(
                            "INSERT INTO attendance (roll_number, date, time, lab) VALUES (?, ?, ?, ?)",
                            (r_num, r_date, r_time, r_lab)
                        )
                conn.commit()
            except Exception as e:
                pass

        # Migrate students from CSV if DB table is empty
        cursor.execute("SELECT COUNT(*) FROM students")
        if cursor.fetchone()[0] == 0 and os.path.exists(STUDENT_REGISTRY_FILE):
            try:
                csv_df = pd.read_csv(STUDENT_REGISTRY_FILE)
                for _, row in csv_df.iterrows():
                    r_num = str(row.get("Roll Number", "")).strip()
                    r_date = str(row.get("Registration Date", "")).strip()
                    r_enc = str(row.get("Face Encoding", "")).strip()
                    r_path = str(row.get("Face Path", "")).strip()
                    if r_num:
                        cursor.execute(
                            "INSERT OR IGNORE INTO students (roll_number, registration_date, face_encoding, face_path) VALUES (?, ?, ?, ?)",
                            (r_num, r_date, r_enc, r_path)
                        )
                conn.commit()
            except Exception as e:
                pass

        # Migrate faculties from CSV if DB table is empty
        cursor.execute("SELECT COUNT(*) FROM faculties")
        if cursor.fetchone()[0] == 0 and os.path.exists(FACULTIES_REGISTRY_FILE):
            try:
                csv_df = pd.read_csv(FACULTIES_REGISTRY_FILE)
                for _, row in csv_df.iterrows():
                    f_name = str(row.get("Name", "")).strip()
                    f_email = str(row.get("Email", "")).strip()
                    f_dept = str(row.get("Department", "")).strip()
                    f_year = str(row.get("Year", "")).strip() if "Year" in row else ""
                    f_sem = str(row.get("Semester", "")).strip() if "Semester" in row else ""
                    if f_name and f_dept:
                        cursor.execute(
                            "INSERT INTO faculties (name, email, department, year, semester) VALUES (?, ?, ?, ?, ?)",
                            (f_name, f_email, f_dept, f_year, f_sem)
                        )
                conn.commit()
            except Exception as e:
                pass

# Initialize SQLite database
init_db()

def load_data():
    with get_db_connection() as conn:
        df = pd.read_sql_query("SELECT roll_number AS 'Roll Number', date AS 'Date', time AS 'Time', lab AS 'Lab' FROM attendance", conn)
    return df

def mark_attendance(roll_number, lab):
    if not roll_number:
        return False, "Roll number cannot be empty."
    
    df = load_data()
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")
    day_of_week = now.weekday()  # 0=Monday, 4=Friday, 5=Saturday
    
    # Check if already marked for this lab today
    if not df[(df["Roll Number"] == roll_number) & (df["Date"] == current_date) & (df["Lab"] == lab)].empty:
        return False, f"Attendance already marked for {roll_number} in {lab} today."
    
    # Get existing attendances for today
    existing_today = df[(df["Roll Number"] == roll_number) & (df["Date"] == current_date)]
    existing_count = len(existing_today)
    
    # On Fridays (4) and Saturdays (5), allow up to 2 attendances, otherwise only 1
    if day_of_week in [4, 6]:  # Friday or Saturday
        max_attendances = 2
    else:
        max_attendances = 1
    
    if existing_count >= max_attendances:
        existing_labs = existing_today["Lab"].unique()
        existing_labs_str = ", ".join(existing_labs)
        return False, f"You can't attend {lab} because you have already reached the maximum attendances ({max_attendances}) for today. Recorded for: {existing_labs_str}."
        
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO attendance (roll_number, date, time, lab) VALUES (?, ?, ?, ?)",
            (str(roll_number), current_date, current_time, lab)
        )
        conn.commit()
    return True, f"Successfully marked attendance for {roll_number} in {lab} at {current_time}."

def load_student_registry():
    with get_db_connection() as conn:
        df = pd.read_sql_query("SELECT roll_number AS 'Roll Number', registration_date AS 'Registration Date', face_encoding AS 'Face Encoding', face_path AS 'Face Path' FROM students", conn)
    if "Face Encoding" not in df.columns:
        df["Face Encoding"] = ""
    if "Face Path" not in df.columns:
        df["Face Path"] = ""
    return df

def save_student_registry(df):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM students")
        for _, row in df.iterrows():
            conn.execute(
                "INSERT OR REPLACE INTO students (roll_number, registration_date, face_encoding, face_path) VALUES (?, ?, ?, ?)",
                (str(row["Roll Number"]), str(row.get("Registration Date", "")), str(row.get("Face Encoding", "")), str(row.get("Face Path", "")))
            )
        conn.commit()

def load_faculties_registry():
    with get_db_connection() as conn:
        df = pd.read_sql_query("SELECT id, name AS 'Name', email AS 'Email', department AS 'Department', year AS 'Year', semester AS 'Semester' FROM faculties", conn)
    for col in ["Year", "Semester"]:
        if col not in df.columns:
            df[col] = ""
    return df


def serialize_face_encoding(encoding):
    """Convert numpy array face encoding to JSON string."""
    if encoding is None:
        return ""
    return json.dumps(encoding.tolist())


def deserialize_face_encoding(encoding_str):
    """Convert JSON string back to numpy array face encoding."""
    if not isinstance(encoding_str, str) or not encoding_str.strip():
        return None
    try:
        return np.array(json.loads(encoding_str))
    except Exception:
        return None


def calculate_ear(eye_points):
    """Calculate Eye Aspect Ratio (EAR) for blink detection.
    Works with both dlib-style tuples and MediaPipe-style numpy arrays."""
    if len(eye_points) < 6:
        return 0.0
    p0, p1, p2, p3, p4, p5 = [np.array(p) for p in eye_points[:6]]
    
    # Distance between vertical eye landmarks
    a = np.linalg.norm(p1 - p5)
    b = np.linalg.norm(p2 - p4)
    
    # Distance between horizontal eye landmarks
    c = np.linalg.norm(p0 - p3)
    
    if c == 0:
        return 0.0
    
    return (a + b) / (2.0 * c)


def calculate_mar(top_lip, bottom_lip):
    """Calculate Mouth Aspect Ratio (MAR) for mouth open detection."""
    if len(top_lip) < 12 or len(bottom_lip) < 12:
        return 0.0
    
    p_left = np.array(top_lip[0])
    p_right = np.array(top_lip[6])
    p_top_inner = np.array(top_lip[9])
    p_bottom_inner = np.array(bottom_lip[9])
    
    width = np.linalg.norm(p_left - p_right)
    vertical = np.linalg.norm(p_top_inner - p_bottom_inner)
    
    if width == 0:
        return 0.0
    
    return vertical / width


def calculate_yaw_ratio(chin, nose_tip):
    """Calculate ratio to estimate head yaw (left/right turning)."""
    if len(chin) < 17 or len(nose_tip) < 5:
        return 1.0
    
    p_left_cheek = chin[0]
    p_right_cheek = chin[16]
    p_nose = nose_tip[2]
    
    d_left = p_nose[0] - p_left_cheek[0]
    d_right = p_right_cheek[0] - p_nose[0]
    
    if d_right == 0:
        return 1.0
        
    return d_left / d_right


def get_landmarks_from_mediapipe(face_landmarks, w, h):
    """Extract landmark point lists from MediaPipe face mesh results.
    Returns dict with keys: left_eye, right_eye, top_lip, bottom_lip, chin, nose_tip."""
    def lm_to_px(idx):
        lm = face_landmarks[idx]
        return (int(lm.x * w), int(lm.y * h))

    return {
        "left_eye": [lm_to_px(i) for i in MP_LEFT_EYE],
        "right_eye": [lm_to_px(i) for i in MP_RIGHT_EYE],
        "top_lip": [lm_to_px(i) for i in MP_TOP_LIP],
        "bottom_lip": [lm_to_px(i) for i in MP_BOTTOM_LIP],
        "chin": [lm_to_px(i) for i in MP_CHIN],
        "nose_tip": [lm_to_px(i) for i in MP_NOSE_TIP],
    }


def match_face(face_image_file, threshold=0.363):
    """
    Given an uploaded or captured image file, detect the face,
    compare it against the registered face encodings in the database,
    and return the matched roll number, or None if no match.
    Uses OpenCV SFace for face embedding/matching.
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return None, "Face recognition library is not available."
    
    try:
        # Load image with PIL and convert to BGR numpy array (OpenCV format)
        image = Image.open(face_image_file).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Extract face embedding using YuNet + SFace
        unknown_embedding, err = extract_face_embedding(image_bgr)
        if unknown_embedding is None:
            return None, err
        
        # Load registry
        df = load_student_registry()
        if df.empty:
            return None, "No students registered in the database."
        
        known_encodings = []
        known_rolls = []
        
        for _, row in df.iterrows():
            enc_str = row.get("Face Encoding", "")
            enc = deserialize_face_encoding(enc_str)
            if enc is not None:
                known_encodings.append(enc)
                known_rolls.append(str(row["Roll Number"]))
                
        if not known_encodings:
            return None, "No registered face encodings found in the database. Please enroll students first."
            
        # Compare faces using SFace cosine similarity
        best_score = -1.0
        best_idx = -1
        for i, known_enc in enumerate(known_encodings):
            is_match, score = compare_face_embeddings(known_enc, unknown_embedding, threshold=threshold)
            if is_match and score > best_score:
                best_score = score
                best_idx = i
        
        if best_idx >= 0:
            return known_rolls[best_idx], None
                
        return None, "Face did not match any registered student in the database."
    except Exception as e:
        return None, f"Error processing face: {str(e)}"



def generate_qr_code_image(roll_number: str):
    """Generate a QR code PIL image for the given roll number."""
    try:
        import qrcode
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(roll_number)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        return img
    except ImportError:
        # Fallback: return a blank white PIL image with text if qrcode not installed
        from PIL import ImageDraw
        img = Image.new("RGB", (200, 200), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 90), roll_number, fill="black")
        return img


def enroll_student(roll_number, face_image_file=None):
    if not roll_number:
        return False, "Roll number cannot be empty."
    df = load_student_registry()
    if str(roll_number) in df["Roll Number"].astype(str).tolist():
        return False, f"Roll number {roll_number} is already registered."
    
    serialized_encoding = ""
    image_path = ""
    if FACE_RECOGNITION_AVAILABLE:
        if face_image_file is None:
            return False, "Face photo is required for enrollment."
        try:
            image = Image.open(face_image_file).convert("RGB")
            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Extract face embedding using YuNet + SFace
            embedding, err = extract_face_embedding(image_bgr)
            if embedding is None:
                return False, err
            serialized_encoding = serialize_face_encoding(embedding)
            
            # Save the face image to the registered_faces folder
            os.makedirs(REGISTERED_FACES_DIR, exist_ok=True)
            image_filename = f"{roll_number}.jpg"
            image_path = os.path.join(REGISTERED_FACES_DIR, image_filename)
            image.save(image_path, "JPEG")
        except Exception as e:
            return False, f"Failed to process face: {str(e)}"
            
    new_record = pd.DataFrame([{
        "Roll Number": roll_number,
        "Registration Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Face Encoding": serialized_encoding,
        "Face Path": image_path
    }])
    df = pd.concat([df, new_record], ignore_index=True)
    save_student_registry(df)
    return True, f"Student {roll_number} registered successfully! QR Code generated below."



def decode_qr_code(image_file):
    """Decode QR code from image file and return the decoded string data"""
    try:
        # Read image
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image_file.seek(0)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return None, "Failed to load image. Make sure it is a valid image format."
        
        detector = cv2.QRCodeDetector()
        data, bbox, straight_qrcode = detector.detectAndDecode(img)
        if data:
            return data, None
        return None, "No QR Code detected in the image."
    except Exception as e:
        return None, f"Error decoding QR Code: {str(e)}"


def auto_scan_qr_from_camera(timeout=12):
    """Scan QR code from camera and return the decoded string data"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, "Unable to open webcam. Please make sure your camera is connected and allowed."
    
    start_ts = time.time()
    detected_data = None
    error_msg = None
    
    # Create a placeholder for the video feed
    stframe = st.empty()
    status_text = st.empty()
    
    detector = cv2.QRCodeDetector()
    
    while time.time() - start_ts < timeout:
        ret, frame = cap.read()
        if not ret:
            error_msg = "Unable to read from webcam."
            break
        
        # Detect and decode QR code
        data, bbox, _ = detector.detectAndDecode(frame)
        
        if data:
            detected_data = data
            # Draw green bounding box around the QR code if found
            if bbox is not None and len(bbox) > 0:
                pts = bbox[0].astype(int)
                for i in range(len(pts)):
                    pt1 = tuple(pts[i])
                    pt2 = tuple(pts[(i + 1) % len(pts)])
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 3)
            
            # Convert to RGB for display
            display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(display_rgb)
            status_text.success(f"✅ QR Code detected: {data}")
            time.sleep(1) # Give user a moment to see the confirmation
            break
        else:
            # Display frame without detection
            display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(display_rgb)
            status_text.info("📷 Scanning... Present your QR Code to the camera.")
        
        time.sleep(0.05)
    
    cap.release()
    stframe.empty()
    status_text.empty()
    
    if detected_data:
        return detected_data, None
    return None, error_msg or "No QR Code detected in the webcam feed."


def auto_scan_face_from_camera(timeout=25, threshold=0.363):
    """Scan webcam video for a registered face, verify liveness via eye blink, and mark attendance.
    Uses MediaPipe FaceMesh for detection/landmarks and OpenCV SFace for recognition."""
    if not FACE_RECOGNITION_AVAILABLE:
        return None, "Face recognition library is not available."

    if not ensure_models_downloaded():
        return None, "Face recognition models could not be loaded."
        
    df = load_student_registry()
    if df.empty:
        return None, "No students registered in the student registry database."
        
    known_encodings = []
    known_rolls = []
    
    for _, row in df.iterrows():
        enc_str = row.get("Face Encoding", "")
        enc = deserialize_face_encoding(enc_str)
        if enc is not None:
            known_encodings.append(enc)
            known_rolls.append(str(row["Roll Number"]))
            
    if not known_encodings:
        return None, "No registered face encodings found. Please enroll students first."
        
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, "Unable to open webcam. Please make sure your camera is connected and allowed."
        
    start_ts = time.time()
    matched_roll = None
    error_msg = None
    
    # Create placeholders
    stframe = st.empty()
    status_text = st.empty()
    
    # Liveness Detection State Variables (Eye Blink Only)
    liveness_verified = False
    
    challenge_timer = time.time()
    max_seconds_per_challenge = 15.0  # Give 15 seconds to blink
    
    face_lost_timestamp = None
    frame_count = 0
    face_match_streak = 0
    required_match_streak = 2
    
    # Blink detection tracking state
    eyes_fully_open = False
    blink_detected = False
    blink_count = 0
    ear_history = []
    
    # Metrics display placeholders
    ear, mar, yaw_ratio = 0.0, 0.0, 1.0
    open_thresh = 0.25
    closed_thresh = 0.20
    
    # Initialize MediaPipe FaceLandmarker (new 1.0 task-based API)
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    landmarker_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_landmarker = FaceLandmarker.create_from_options(landmarker_options)
    recognizer = get_face_recognizer()
    
    # Recognition throttle — only run SFace every N frames
    recognition_interval = 5
    
    while time.time() - start_ts < timeout:
        ret, frame = cap.read()
        if not ret:
            error_msg = "Unable to read from webcam."
            break
            
        frame_count += 1
        
        # Mirror the frame horizontally for intuitive self-viewing
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run MediaPipe FaceLandmarker (task-based API)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start_ts) * 1000)
        results = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if not results.face_landmarks:
            # Face lost tracking
            if face_lost_timestamp is None:
                face_lost_timestamp = time.time()
            elif time.time() - face_lost_timestamp > 4.0:
                # Reset identification and challenge state only after several seconds of loss
                matched_roll = None
                blink_detected = False
                eyes_fully_open = False
                ear_history = []
                challenge_timer = time.time()
        else:
            # Face found, reset face lost timestamp
            face_lost_timestamp = None
            
            face_lm = results.face_landmarks[0]
            
            # Extract landmarks for EAR/MAR/yaw
            landmarks = get_landmarks_from_mediapipe(face_lm, w, h)
            left_eye = landmarks["left_eye"]
            right_eye = landmarks["right_eye"]
            top_lip = landmarks["top_lip"]
            bottom_lip = landmarks["bottom_lip"]
            chin = landmarks["chin"]
            nose_tip = landmarks["nose_tip"]
            
            # Calculate bounding box from all landmarks
            all_x = [int(lm.x * w) for lm in face_lm]
            all_y = [int(lm.y * h) for lm in face_lm]
            left = max(0, min(all_x) - 10)
            top = max(0, min(all_y) - 10)
            right = min(w, max(all_x) + 10)
            bottom = min(h, max(all_y) + 10)
            
            # Calculate real-time metrics
            ear_l = calculate_ear(left_eye)
            ear_r = calculate_ear(right_eye)
            ear = (ear_l + ear_r) / 2.0
            mar = calculate_mar(top_lip, bottom_lip)
            yaw_ratio = calculate_yaw_ratio(chin, nose_tip)
            
            # --- STATE MACHINE ---
            if matched_roll is None:
                # Phase 1: Identify Face (throttled to every N frames for performance)
                if frame_count % recognition_interval == 0:
                    try:
                        detector = get_face_detector(w, h)
                        _, faces = detector.detect(frame)
                        if faces is not None and len(faces) > 0:
                            aligned_face = recognizer.alignCrop(frame, faces[0])
                            face_emb = recognizer.feature(aligned_face).flatten()
                            
                            best_score = -1.0
                            best_idx = -1
                            for i, known_enc in enumerate(known_encodings):
                                is_match, score = compare_face_embeddings(known_enc, face_emb, threshold=threshold)
                                if is_match and score > best_score:
                                    best_score = score
                                    best_idx = i
                            
                            if best_idx >= 0:
                                matched_roll = known_rolls[best_idx]
                                challenge_timer = time.time()
                                blink_detected = False
                                eyes_fully_open = False
                                ear_history = []
                    except Exception:
                        pass  # Skip recognition errors, try again next interval
                        
            elif not liveness_verified:
                # Phase 2: Verify Liveness via blink detection
                elapsed = time.time() - challenge_timer
                if elapsed > max_seconds_per_challenge:
                    matched_roll = None
                    blink_detected = False
                    eyes_fully_open = False
                    ear_history = []
                    challenge_timer = time.time()
                    continue

                ear_history.append(ear)
                if len(ear_history) > 25:
                    ear_history.pop(0)

                if len(ear_history) >= 10:
                    sorted_ear = sorted(ear_history)
                    base_ear = sorted_ear[int(len(sorted_ear) * 0.8)]
                    open_thresh = min(max(0.20, base_ear * 0.90), 0.32)
                    closed_thresh = max(min(0.25, base_ear * 0.80), 0.14)
                else:
                    open_thresh = 0.24
                    closed_thresh = 0.19

                if ear > open_thresh:
                    eyes_fully_open = True
                if eyes_fully_open and ear < closed_thresh:
                    blink_detected = True
                    blink_count += 1

                reopen_thresh = max(open_thresh - 0.08, closed_thresh + 0.02)
                if blink_detected and ear > reopen_thresh:
                    liveness_verified = True
                    blink_detected = False
                    blink_count = 0

                if not liveness_verified and len(ear_history) >= 8:
                    min_val = min(ear_history)
                    min_idx = ear_history.index(min_val)
                    max_before = max(ear_history[:min_idx]) if min_idx > 0 else min_val
                    max_after = max(ear_history[min_idx+1:]) if min_idx < len(ear_history) - 1 else min_val
                    min_open = max(0.22, open_thresh * 0.88)
                    if (max_before - min_val) > 0.022 and (max_after - min_val) > 0.022 and min_val < min_open:
                        liveness_verified = True
            
            # --- DRAW HUD ON FRAME ---
            # Set bounding box color depending on state
            if liveness_verified:
                box_color = (0, 255, 0)  # Green
            elif matched_roll is not None:
                box_color = (0, 165, 255)  # Orange/Gold
            else:
                box_color = (255, 255, 255)  # White
                
            # Draw face box
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            
            # Draw telemetry landmarks (dots) on eyes to look high-tech
            for eye_pt in left_eye + right_eye:
                cv2.circle(frame, eye_pt, 2, (0, 255, 255), -1)  # Yellow dots on eyes
                    
        # --- SEMI-TRANSPARENT TOP & BOTTOM HUD BANDS ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 55), (15, 15, 15), -1)
        cv2.rectangle(overlay, (0, frame.shape[0] - 90), (frame.shape[1], frame.shape[0]), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        
        # Write Title and Status on Top Bar
        if liveness_verified:
            status_title = "LIVENESS VERIFIED"
            status_color = (0, 255, 0)  # Green
        elif matched_roll is not None:
            status_title = f"IDENTIFIED: STUDENT {matched_roll}"
            status_color = (0, 165, 255)  # Orange
        else:
            status_title = "SCANNING FOR REGISTERED FACE..."
            status_color = (255, 255, 0)  # Cyan
            
        cv2.putText(frame, status_title, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)
        
        # Write Live Telemetry on the Right Side
        cv2.putText(frame, f"EAR: {ear:.2f}", (frame.shape[1] - 110, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"MAR: {mar:.2f}", (frame.shape[1] - 110, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"YAW: {yaw_ratio:.2f}", (frame.shape[1] - 110, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Write Challenge Checklist on Bottom Bar
        if matched_roll is None:
            cv2.putText(frame, "Position your face clearly in the camera feed.", (15, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, "Verification will begin automatically upon matching.", (15, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        else:
            # Challenges status
            c_status = "[ OK ]" if liveness_verified else f"[ACTIVE ({max_seconds_per_challenge - (time.time() - challenge_timer):.1f}s)]"
            c_color = (0, 255, 0) if liveness_verified else (0, 165, 255)
            
            cv2.putText(frame, f"Liveness Check: Blink your eyes -> {c_status}", (15, frame.shape[0] - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c_color, 1, cv2.LINE_AA)
            cv2.putText(frame, f"EAR Telemetry: current={ear:.2f} | target closed<{closed_thresh:.2f}", (15, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
            
            if not liveness_verified and blink_detected:
                cv2.putText(frame, "Eyes closed... Open them!", (frame.shape[1] - 200, frame.shape[0] - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                
        # Display the frame in Streamlit
        display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(display_rgb)
        
        # Display messages in Streamlit status_text
        if liveness_verified:
            status_text.success(f"✅ Match & Liveness Verified: Student {matched_roll}")
            time.sleep(2.0)
            break
        elif matched_roll is not None:
            status_text.warning("🔒 Liveness Challenge: Please blink your eyes!")
        else:
            status_text.info("📷 Scanning... Position your face clearly in front of the camera.")
            
        time.sleep(0.01)
        
    face_landmarker.close()
    cap.release()
    stframe.empty()
    status_text.empty()
    
    if liveness_verified and matched_roll:
        return matched_roll, None
    return None, error_msg or "Liveness verification failed or timed out."


def play_siri_voice(success, roll_number=""):
    """Play siri voice note based on success/failure"""
    try:
        import platform
        import os
        system = platform.system()
        if system == 'Darwin':  # macOS
            roll_str = str(roll_number).replace('"', '').replace("'", "")
            prefix = f"{roll_str}, " if roll_str else ""
            if success:
                os.system(f'say -v siri "{prefix}your attendance is successfully recorded." &')
            else:
                os.system(f'say -v siri "{prefix}failed to record attendance." ')
    except Exception:
        pass

# Initialize session state for popup
if 'show_popup' not in st.session_state:
    st.session_state.show_popup = False
if 'popup_data' not in st.session_state:
    st.session_state.popup_data = {}

def show_scan_popup(success, message, roll_number=""):
    """Display a popup after scanning"""
    st.session_state.show_popup = True
    st.session_state.popup_data = {
        'success': success,
        'message': message,
        'roll_number': roll_number
    }

def render_popup():
    """Render the popup modal"""
    if st.session_state.show_popup:
        popup_data = st.session_state.popup_data
        icon = "✅" if popup_data['success'] else "❌"
        status_class = "modal-success" if popup_data['success'] else "modal-error"
        heading = "Attendance Marked!" if popup_data['success'] else "Failed to Mark"
        
        st.markdown(f"""
        <div class="modal-overlay">
            <div class="modal-content">
                <div class="modal-icon {status_class}">{icon}</div>
                <h2>{heading}</h2>
                <div class="modal-message">
                    {popup_data['message']}
                </div>
                {f'<div class="modal-footer">Roll: <strong>{popup_data["roll_number"]}</strong></div>' if popup_data.get('roll_number') else ''}
            </div>
        </div>
        <script>
            setTimeout(() => {{
                //Auto close popup after 3 seconds
                let overlay = document.querySelector('.modal-overlay');
                if (overlay) overlay.style.display = 'none';
            }}, 3000);
        </script>
        """, unsafe_allow_html=True)
        
        # Reset popup state so it doesn't reappear on other interactions
        st.session_state.show_popup = False

# ------------------ LOGIN & ROUTING REDIRECT LOGIC ------------------

if st.session_state.redirecting:
    role = st.session_state.target_role
    username = st.session_state.username
    role_title = "Student" if role == "student" else "Faculty"
    
    st.markdown(
        f"""
        <div class="login-page" style="min-height: 80vh;">
            <div class="login-card" style="text-align: center;">
                <div style="font-size: 3.5rem; margin-bottom: 12px; filter: drop-shadow(var(--glow-indigo));">🔑</div>
                <h2>Signing in...</h2>
                <p style="color: var(--text-secondary); margin: 8px 0 0 0; font-size: 0.95rem;">Welcome — redirecting {role_title} to attendance page...</p>
                <div class="loader-container">
                    <div class="futuristic-loader"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    time.sleep(1.5)
    st.session_state.logged_in = True
    st.session_state.user_role = role
    st.session_state.redirecting = False
    st.rerun()

else:
    # Student direct access or faculty wants to login
    col_logo, col_faculty_btn, col_admin_btn = st.columns([4, 1, 1])
    with col_logo:
        st.markdown("<h1 style='margin:0; font-size: 1.6rem; font-weight: 600; color: #f4f4f5;'>Smart Lab Attendance</h1>", unsafe_allow_html=True)
    with col_faculty_btn:
        st.markdown("<div style='padding-top: 4px;'></div>", unsafe_allow_html=True)
        if st.session_state.user_role == "student":
            if st.button("Faculty Login", key="faculty_login_nav_btn", use_container_width=True):
                st.session_state.show_faculty_login = True
                st.session_state.show_admin_login = False
                st.rerun()
        else:
            if st.button("Logout", key="logout_btn", use_container_width=True):
                st.session_state.user_role = "student"
                st.session_state.username = "Student"
                st.rerun()
    with col_admin_btn:
        st.markdown("<div style='padding-top: 4px;'></div>", unsafe_allow_html=True)
        if st.session_state.user_role == "student":
            if st.button("Admin Login", key="admin_login_nav_btn", use_container_width=True):
                st.session_state.show_admin_login = True
                st.session_state.show_faculty_login = False
                st.rerun()

    # Faculty login modal if toggled
    if st.session_state.get("show_faculty_login", False):
        st.markdown("---")
        st.subheader("🔐 Faculty Sign In")
        faculty_email = st.text_input("Faculty Email", placeholder="professor@college.edu", key="faculty_email_input")
        faculty_pw = st.text_input("Password", type="password", placeholder="••••••••", key="faculty_pw_input")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Sign In as Faculty", use_container_width=True, key="faculty_login_btn"):
                if not faculty_email.strip():
                    st.error("Please enter a valid Email.")
                elif faculty_pw != "faculty123":
                    st.error("Incorrect password. Hint: faculty123")
                else:
                    st.session_state.user_role = "faculty"
                    st.session_state.username = faculty_email.strip()
                    st.session_state.show_faculty_login = False
                    st.rerun()
        with col2:
            if st.button("Cancel", use_container_width=True, key="cancel_faculty_login"):
                st.session_state.show_faculty_login = False
                st.rerun()
        
        st.markdown("---")
            
    # Admin login modal if toggled
    if st.session_state.get("show_admin_login", False):
        st.markdown("---")
        st.subheader("👑 Admin Sign In")
        admin_id = st.text_input("Admin ID", placeholder="admin", key="admin_id_input")
        admin_pw = st.text_input("Password", type="password", placeholder="••••••••", key="admin_pw_input")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Sign In as Admin", use_container_width=True, key="admin_login_btn"):
                if not admin_id.strip():
                    st.error("Please enter Admin ID.")
                elif admin_pw != "admin123":
                    st.error("Incorrect password. Hint: admin123")
                else:
                    st.session_state.user_role = "admin"
                    st.session_state.username = admin_id.strip()
                    st.session_state.show_admin_login = False
                    st.rerun()
        with col2:
            if st.button("Cancel", use_container_width=True, key="cancel_admin_login"):
                st.session_state.show_admin_login = False
                st.rerun()
        
        st.markdown("---")
            
    st.markdown(
        """
        <div class="hero" style="margin-top: 15px;">
            <h2>Secure Attendance Dashboard</h2>
            <p>Scan QR codes using your camera feed, manage student registers, and export reports directly from your secure session.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Summary metrics
    summary_df = load_data()
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_count = summary_df[summary_df["Date"] == today_str].shape[0]
    unique_students = summary_df["Roll Number"].nunique()
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Records", f"{len(summary_df)}")
    col_b.metric("Today's Marks", f"{today_count}")
    col_c.metric("Unique Students", f"{unique_students}")
    
    st.divider()    # Configure tabs based on user role
    if st.session_state.user_role == "admin":
        tab_list = ["👨‍🏫 Manage Faculty", "📥 Download Attendance", "📊 Attendance Analytics", "🗑️ Manage Records"]
    elif st.session_state.user_role == "student":
        tab_list = ["👤 Face & QR Attendance", "👤 Register My Face", "📇 My QR Code"]
    else:
        tab_list = ["👤 Face & QR Attendance", "📊 View Records", "🧑‍🎓 Manage Students & Faces", "📥 Download Student Record"]
        
    tabs = st.tabs(tab_list)
    
    # Define subjects hierarchy (used across admin tabs)
    SUBJECTS_BY_YEAR_SEM = {
        "1st Year": {
            "1st Sem": ["C Programing", "DigitalLogics"],
            "2nd Sem": ["Cpp", "DataStructure"]
        },
        "2nd Year": {
            "3rd Sem": ["DBMS", "JAVA", "WebDesigen"],
            "4th Sem": ["Python", "Operating System", "Computer Graphics"]
        },
        "3rd Year": {
            "5th Sem": ["PHP", "DAA", "C#"],
            "6th Sem": ["AIML"]
        }
    }

    ALL_YEARS = list(SUBJECTS_BY_YEAR_SEM.keys())
    ALL_SEMESTERS = {yr: list(sems.keys()) for yr, sems in SUBJECTS_BY_YEAR_SEM.items()}

    if st.session_state.user_role == "admin":
        # ==================== TAB 0: MANAGE FACULTY ====================
        with tabs[0]:
            st.header("👨‍🏫 Manage Faculty")
            st.markdown("<div class='section-note'>Add new faculty members with their year, semester, and subject assignments. Edit or remove existing entries.</div>", unsafe_allow_html=True)
            fac_df = load_faculties_registry()

            with st.form("add_faculty_form"):
                st.subheader("➕ Add New Faculty")
                f_name = st.text_input("Faculty Name", placeholder="e.g. Dr. John Doe")
                f_email = st.text_input("Faculty Email", placeholder="e.g. john@college.edu")

                f_year = st.selectbox("Year", ALL_YEARS, key="add_fac_year")
                available_sems = ALL_SEMESTERS.get(f_year, [])
                f_sem = st.selectbox("Semester", available_sems if available_sems else ["N/A"], key="add_fac_sem")

                # Get subjects for the selected year/semester
                available_subjects = SUBJECTS_BY_YEAR_SEM.get(f_year, {}).get(f_sem, [])
                if available_subjects:
                    f_dept = st.selectbox("Subject", available_subjects, key="add_fac_subject")
                else:
                    f_dept = st.text_input("Subject (no predefined subjects for this semester)", placeholder="e.g. Python, DBMS", key="add_fac_subject_manual")

                submitted = st.form_submit_button("➕ Add Faculty", type="primary")
                if submitted:
                    if f_name and f_email and f_dept:
                        # Check for duplicate: same name + same subject + same semester
                        dup = fac_df[
                            (fac_df["Name"].fillna("").astype(str).str.strip().str.lower() == f_name.strip().lower()) &
                            (fac_df["Department"].fillna("").astype(str).str.strip().str.lower() == str(f_dept).strip().lower()) &
                            (fac_df["Semester"].fillna("").astype(str).str.strip().str.lower() == str(f_sem).strip().lower())
                        ]
                        if not dup.empty:
                            st.error(f"⚠️ {f_name} is already assigned to {f_dept} in {f_sem}. Duplicate entry blocked.")
                        else:
                            with get_db_connection() as conn:
                                conn.execute(
                                    "INSERT INTO faculties (name, email, department, year, semester) VALUES (?, ?, ?, ?, ?)",
                                    (f_name.strip(), f_email.strip(), str(f_dept).strip(), f_year, f_sem)
                                )
                                conn.commit()
                            st.success(f"✅ Successfully added **{f_name}** → {f_dept} ({f_year} / {f_sem})")
                            st.rerun()
                    else:
                        st.error("Please fill in all fields.")

            st.divider()
            st.subheader("📋 Registered Faculties")
            if fac_df.empty:
                st.info("No faculties registered yet. Use the form above to add your first faculty.")
            else:
                st.dataframe(
                    fac_df[["Name", "Email", "Department", "Year", "Semester"]],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Name": st.column_config.TextColumn("Faculty Name", width="medium"),
                        "Email": st.column_config.TextColumn("Email", width="medium"),
                        "Department": st.column_config.TextColumn("Subject", width="medium"),
                        "Year": st.column_config.TextColumn("Year", width="small"),
                        "Semester": st.column_config.TextColumn("Semester", width="small"),
                    }
                )

                # Delete faculty option
                st.subheader("🗑️ Remove Faculty")
                fac_options = fac_df.apply(
                    lambda r: f"{r['Name']} — {r.get('Department', '')} ({r.get('Year', '')} / {r.get('Semester', '')})",
                    axis=1
                ).tolist()
                fac_to_delete = st.selectbox("Select faculty to remove:", ["-- Select --"] + fac_options, key="del_fac_select")
                if fac_to_delete != "-- Select --":
                    idx_to_del = fac_options.index(fac_to_delete)
                    if st.button("🗑️ Delete Selected Faculty", type="primary", key="del_fac_btn"):
                        sel_row = fac_df.iloc[idx_to_del]
                        fac_id = sel_row.get("id")
                        with get_db_connection() as conn:
                            if fac_id and not pd.isna(fac_id):
                                conn.execute("DELETE FROM faculties WHERE id = ?", (int(fac_id),))
                            else:
                                conn.execute("DELETE FROM faculties WHERE name = ? AND department = ?", (str(sel_row["Name"]), str(sel_row["Department"])))
                            conn.commit()
                        st.success(f"✅ Removed: {fac_to_delete}")
                        st.rerun()

        # ==================== TAB 1: DOWNLOAD ATTENDANCE ====================
        with tabs[1]:
            st.header("📥 Download Attendance Records")
            st.markdown("<div class='section-note'>Download attendance records <strong>per faculty</strong>, filtered by <strong>subject</strong> and <strong>semester</strong>. Each faculty's data can be downloaded separately as CSV, or download everything as a ZIP.</div>", unsafe_allow_html=True)

            fac_df = load_faculties_registry()
            att_df = load_data()

            if fac_df.empty:
                st.warning("⚠️ No faculties registered yet. Go to **Manage Faculty** tab to add faculties first.")
            elif att_df.empty:
                st.warning("⚠️ No attendance records found yet.")
            else:
                # --- Filters Section ---
                st.subheader("🔍 Filter Options")

                filter_col1, filter_col2, filter_col3 = st.columns(3)
                with filter_col1:
                    dl_year_filter = st.selectbox("Filter by Year", ["All Years"] + ALL_YEARS, key="admin_dl_year")
                with filter_col2:
                    if dl_year_filter != "All Years":
                        sem_options = ALL_SEMESTERS.get(dl_year_filter, [])
                    else:
                        sem_options = sorted(set(s for sems in ALL_SEMESTERS.values() for s in sems))
                    dl_sem_filter = st.selectbox("Filter by Semester", ["All Semesters"] + sem_options, key="admin_dl_sem")
                with filter_col3:
                    fac_names_unique = sorted(fac_df["Name"].dropna().unique().tolist())
                    dl_fac_filter = st.selectbox("Filter by Faculty", ["All Faculties"] + fac_names_unique, key="admin_dl_fac")

                col_start, col_end = st.columns(2)
                with col_start:
                    start_date = st.date_input(
                        "Start Date",
                        value=pd.to_datetime(att_df["Date"].min()).date() if not att_df.empty else datetime.now().date(),
                        key="admin_dl_start_date"
                    )
                with col_end:
                    end_date = st.date_input(
                        "End Date",
                        value=datetime.now().date(),
                        key="admin_dl_end_date"
                    )

                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")

                # Filter attendance by date range
                date_filtered_att = att_df[(att_df["Date"] >= start_str) & (att_df["Date"] <= end_str)]

                # Apply filters to faculty list
                filtered_fac = fac_df.copy()
                for _col in ["Year", "Semester", "Name", "Department"]:
                    if _col in filtered_fac.columns:
                        filtered_fac[_col] = filtered_fac[_col].fillna("").astype(str)
                if dl_year_filter != "All Years":
                    filtered_fac = filtered_fac[filtered_fac["Year"].str.strip() == dl_year_filter]
                if dl_sem_filter != "All Semesters":
                    filtered_fac = filtered_fac[filtered_fac["Semester"].str.strip() == dl_sem_filter]
                if dl_fac_filter != "All Faculties":
                    filtered_fac = filtered_fac[filtered_fac["Name"].str.strip() == dl_fac_filter]

                if date_filtered_att.empty:
                    st.info(f"No attendance records found between {start_date.strftime('%b %d, %Y')} and {end_date.strftime('%b %d, %Y')}.")
                elif filtered_fac.empty:
                    st.info("No faculties match the selected filters.")
                else:
                    # Count matching records
                    total_matching = 0
                    for _, fac_row in filtered_fac.iterrows():
                        fac_subject = str(fac_row.get("Department", "")).strip()
                        if fac_subject:
                            subject_att = date_filtered_att[date_filtered_att["Lab"].str.strip().str.lower() == fac_subject.lower()]
                            total_matching += len(subject_att)

                    st.success(f"📋 Found **{total_matching}** records across **{len(filtered_fac)}** faculty entries — **{start_date.strftime('%b %d, %Y')}** to **{end_date.strftime('%b %d, %Y')}**")

                    st.divider()

                    # --- Bulk Download ZIP ---
                    st.subheader("📦 Bulk Download (ZIP)")
                    st.markdown("Download all filtered faculty attendance data as a single ZIP file, organized by semester.")

                    import zipfile
                    zip_buffer = BytesIO()
                    has_data = False
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        for _, fac_row in filtered_fac.iterrows():
                            fac_subject = str(fac_row.get("Department", "")).strip()
                            fac_name = str(fac_row.get("Name", "")).strip()
                            fac_sem = str(fac_row.get("Semester", "")).strip()
                            fac_year = str(fac_row.get("Year", "")).strip()
                            if not fac_subject:
                                continue
                            subject_data = date_filtered_att[date_filtered_att["Lab"].str.strip().str.lower() == fac_subject.lower()]
                            if not subject_data.empty:
                                has_data = True
                                csv_content = subject_data[["Roll Number", "Date", "Time", "Lab"]].to_csv(index=False)
                                safe_name = fac_name.replace(" ", "_") if fac_name else fac_subject.replace(" ", "_")
                                safe_sem = fac_sem.replace(" ", "_") if fac_sem else "unknown_sem"
                                folder = f"{fac_year.replace(' ', '_')}/{safe_sem}" if fac_year else safe_sem
                                zf.writestr(f"{folder}/{safe_name}_{fac_subject}_attendance.csv", csv_content)

                    if has_data:
                        st.download_button(
                            label="⬇️ Download All Filtered Attendance (ZIP)",
                            data=zip_buffer.getvalue(),
                            file_name=f"faculty_attendance_{start_str}_to_{end_str}.zip",
                            mime="application/zip",
                            key="admin_dl_all_zip",
                            use_container_width=True,
                        )
                    else:
                        st.info("No matching attendance data found for the filtered faculty subjects.")

                    st.divider()

                    # --- Per-Faculty Download Cards (grouped by semester) ---
                    st.subheader("👨‍🏫 Download Per Faculty / Semester")

                    # Group filtered faculties by Year → Semester
                    grouped_by_sem = {}
                    for _, fac_row in filtered_fac.iterrows():
                        yr = str(fac_row.get("Year", "Unspecified")).strip() or "Unspecified"
                        sm = str(fac_row.get("Semester", "Unspecified")).strip() or "Unspecified"
                        key = f"{yr} — {sm}"
                        if key not in grouped_by_sem:
                            grouped_by_sem[key] = []
                        grouped_by_sem[key].append(fac_row)

                    for group_label, fac_rows in grouped_by_sem.items():
                        st.markdown(f"#### 📅 {group_label}")

                        for fac_row in fac_rows:
                            fac_idx = fac_row.name if hasattr(fac_row, 'name') else 0
                            fac_name = str(fac_row.get("Name", "Unknown")).strip()
                            fac_email = str(fac_row.get("Email", "")).strip()
                            fac_subject = str(fac_row.get("Department", "")).strip()
                            fac_year = str(fac_row.get("Year", "")).strip()
                            fac_sem = str(fac_row.get("Semester", "")).strip()

                            if not fac_subject:
                                continue

                            # Filter attendance for this faculty's subject
                            subject_att = date_filtered_att[date_filtered_att["Lab"].str.strip().str.lower() == fac_subject.lower()]

                            with st.expander(f"👤 {fac_name} — 📚 {fac_subject} — 🗓️ {fac_sem} ({len(subject_att)} records)", expanded=False):
                                st.markdown(f"""
                                | Field | Value |
                                |-------|-------|
                                | **Faculty Name** | {fac_name} |
                                | **Email** | {fac_email} |
                                | **Subject** | {fac_subject} |
                                | **Year** | {fac_year} |
                                | **Semester** | {fac_sem} |
                                | **Total Records** | {len(subject_att)} |
                                | **Unique Students** | {subject_att['Roll Number'].nunique() if not subject_att.empty else 0} |
                                | **Date Range** | {start_date.strftime('%b %d, %Y')} → {end_date.strftime('%b %d, %Y')} |
                                """)

                                if subject_att.empty:
                                    st.info(f"No attendance records found for **{fac_subject}** in the selected date range.")
                                else:
                                    st.dataframe(
                                        subject_att[["Roll Number", "Date", "Time", "Lab"]].sort_values(["Date", "Time"], ascending=[False, False]),
                                        use_container_width=True,
                                        hide_index=True,
                                        column_config={
                                            "Roll Number": st.column_config.TextColumn("Student Roll Number", width="medium"),
                                            "Date": st.column_config.TextColumn("Date", width="small"),
                                            "Time": st.column_config.TextColumn("Time", width="small"),
                                            "Lab": st.column_config.TextColumn("Subject", width="medium"),
                                        }
                                    )

                                    m1, m2, m3 = st.columns(3)
                                    with m1:
                                        st.metric("Total Records", len(subject_att))
                                    with m2:
                                        st.metric("Unique Students", subject_att["Roll Number"].nunique())
                                    with m3:
                                        st.metric("Active Days", subject_att["Date"].nunique())

                                    csv_data = subject_att[["Roll Number", "Date", "Time", "Lab"]].to_csv(index=False).encode("utf-8")
                                    safe_fname = fac_name.replace(" ", "_")
                                    safe_sem_name = fac_sem.replace(" ", "_") if fac_sem else "all"
                                    st.download_button(
                                        label=f"⬇️ Download {fac_subject} — {fac_sem} Attendance CSV",
                                        data=csv_data,
                                        file_name=f"{safe_fname}_{fac_subject}_{safe_sem_name}_{start_str}_to_{end_str}.csv",
                                        mime="text/csv",
                                        key=f"admin_dl_fac_{fac_idx}_{fac_subject}_{safe_sem_name}",
                                        use_container_width=True,
                                    )

                        st.markdown("---")

                    # --- Summary Table ---
                    st.subheader("📊 Faculty Summary Table")
                    summary_rows = []
                    for _, fac_row in filtered_fac.iterrows():
                        fac_name = str(fac_row.get("Name", "Unknown")).strip()
                        fac_subject = str(fac_row.get("Department", "")).strip()
                        fac_year = str(fac_row.get("Year", "")).strip()
                        fac_sem = str(fac_row.get("Semester", "")).strip()
                        if not fac_subject:
                            continue
                        subject_att = date_filtered_att[date_filtered_att["Lab"].str.strip().str.lower() == fac_subject.lower()]
                        summary_rows.append({
                            "Faculty": fac_name,
                            "Subject": fac_subject,
                            "Year": fac_year,
                            "Semester": fac_sem,
                            "Total Records": len(subject_att),
                            "Unique Students": subject_att["Roll Number"].nunique() if not subject_att.empty else 0,
                            "Active Days": subject_att["Date"].nunique() if not subject_att.empty else 0,
                        })

                    if summary_rows:
                        summary_table = pd.DataFrame(summary_rows)
                        st.dataframe(summary_table, use_container_width=True, hide_index=True)
                    else:
                        st.info("No faculty subjects to summarize.")

        # ==================== TAB 2: ATTENDANCE ANALYTICS ====================
        with tabs[2]:
            st.header("📊 Attendance Analytics")
            st.markdown("<div class='section-note'>Overview of student attendance trends.</div>", unsafe_allow_html=True)
            att_df = load_data()
            if not att_df.empty:
                st.subheader("📈 Attendance Over Time")
                date_counts = att_df.groupby("Date").size().reset_index(name="Total Attendance")
                st.line_chart(date_counts.set_index("Date"), use_container_width=True)

                st.divider()

                st.subheader("📚 Attendance by Lab Subject")
                lab_counts = att_df.groupby("Lab").size().reset_index(name="Total Attendance")
                st.bar_chart(lab_counts.set_index("Lab"), use_container_width=True)
            else:
                st.info("No attendance records found yet.")

        # ==================== TAB 3: MANAGE RECORDS (Delete etc.) ====================
        with tabs[3]:
            st.header("🗑️ Manage Attendance Records")
            st.markdown("<div class='section-note'>View, search, and delete attendance records. Use with caution — deletions are permanent.</div>", unsafe_allow_html=True)

            att_df_manage = load_data()
            if att_df_manage.empty:
                st.info("No attendance records found.")
            else:
                manage_date = st.date_input("Select Date", value=datetime.now().date(), key="admin_manage_date")
                manage_date_str = manage_date.strftime("%Y-%m-%d")
                day_records = att_df_manage[att_df_manage["Date"] == manage_date_str]

                if day_records.empty:
                    st.info(f"No records for {manage_date.strftime('%B %d, %Y')}.")
                else:
                    st.dataframe(day_records, use_container_width=True, hide_index=True)

                    st.subheader("Delete Individual Record")
                    rec_options = day_records.apply(lambda r: f"{r['Roll Number']} — {r['Lab']} — {r['Time']}", axis=1).tolist()
                    rec_indices = day_records.index.tolist()
                    rec_map = dict(zip(rec_options, rec_indices))
                    rec_to_del = st.selectbox("Select record:", ["-- Select --"] + rec_options, key="admin_del_rec")
                    if rec_to_del != "-- Select --":
                        if st.button("🗑️ Delete Record", type="primary", key="admin_del_rec_btn"):
                            sel_rec = day_records.loc[rec_map[rec_to_del]]
                            with get_db_connection() as conn:
                                conn.execute(
                                    "DELETE FROM attendance WHERE roll_number = ? AND date = ? AND time = ? AND lab = ?",
                                    (str(sel_rec["Roll Number"]), str(sel_rec["Date"]), str(sel_rec["Time"]), str(sel_rec["Lab"]))
                                )
                                conn.commit()
                            st.success(f"✅ Deleted: {rec_to_del}")
                            st.rerun()

                    st.divider()
                    if st.button(f"🗑️ Delete ALL records for {manage_date.strftime('%b %d, %Y')}", key="admin_del_day_btn"):
                        with get_db_connection() as conn:
                            conn.execute("DELETE FROM attendance WHERE date = ?", (manage_date_str,))
                            conn.commit()
                        st.success(f"✅ Deleted all records for {manage_date.strftime('%b %d, %Y')}")
                        st.rerun()

        st.stop()
    
    with tabs[0]:
        st.header("👤 Face & QR Scanner Attendance")
        # Initialize selected_subject in session state if not already set
        if 'selected_subject' not in st.session_state:
            st.session_state.selected_subject = "C Programing"

        # Define subjects hierarchy
        subjects_by_year_sem = {
            "1st Year": {
                "1st Sem": ["C Programing", "DigitalLogics",],
                "2nd Sem": ["Cpp", "DataStructure"]
            },
            "2nd Year": {
                "3rd Sem": ["DBMS", "JAVA", "WebDesigen"],
                "4th Sem": ["Python", "Operating System", "Computer Graphics"]
            },
            "3rd Year": {
                "5th Sem": ["PHP","DAA","C#",],
                "6th Sem": ["AIML"]
            }
        }

        st.markdown('<div class="subject-selection-marker"></div>', unsafe_allow_html=True)

        # Dynamic accordion: expand only the year that contains the selected subject
        active_year = "1st Year"
        for yr, sems in subjects_by_year_sem.items():
            for sem, subs in sems.items():
                if st.session_state.selected_subject in subs:
                    active_year = yr

        for yr in ["1st Year", "2nd Year", "3rd Year"]:
            is_expanded = (yr == active_year)
            with st.expander(f"📅 {yr}", expanded=is_expanded):
                sems = list(subjects_by_year_sem[yr].keys())
                sem_cols = st.columns(len(sems))
                for s_idx, sem in enumerate(sems):
                    with sem_cols[s_idx]:
                        st.markdown(f"<div style='font-size: 0.95rem; color: #222222; margin-top: 4px; margin-bottom: 8px; font-weight: 600;'>{sem}</div>", unsafe_allow_html=True)
                        subs = subjects_by_year_sem[yr][sem]
                        for sub in subs:
                            if not sub:
                                st.markdown("<p style='color:#888;margin:0;font-size:0.875rem;'>No subjects added yet</p>", unsafe_allow_html=True)
                                continue
                            is_selected = (st.session_state.selected_subject == sub)
                            btn_label = f"✅  {sub}" if is_selected else f"📄  {sub}"
                            if st.button(
                                btn_label, 
                                key=f"btn_sel_{yr}_{sem}_{sub}", 
                                use_container_width=True, 
                                type="primary" if is_selected else "secondary"
                            ):
                                st.session_state.selected_subject = sub
                                st.rerun()

        lab_choice = st.session_state.selected_subject
        st.markdown("<div class='section-note'>Select the correct lab subject first, then use your face or QR Code to verify and mark attendance.</div>", unsafe_allow_html=True)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            method = st.radio("Verification Method:", ["Face Recognition", "QR Code Scanner"], horizontal=True, key="attendance_method_radio")
            
            if method == "Face Recognition":
                st.subheader("👤 Secure Face Attendance Scanner")
                if not FACE_RECOGNITION_AVAILABLE:
                    st.warning("⚠️ Face recognition is not available. Please install 'mediapipe' library or use QR Code mode.")
                else:
                    st.markdown("""
                    <div class='section-note' style='border-left-color: #10b981; background: rgba(16, 185, 129, 0.05); margin-bottom: 15px; padding: 12px; border-radius: 8px;'>
                        <strong>🛡️ Liveness Detection Protocol:</strong> The face recognition scanner requires you to blink your eyes to verify that you are a live person. Static photos or device screen displays will be rejected.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("📷 Start Live Face Scanner", key="start_live_face_scanner_btn", type="primary", use_container_width=True):
                        st.info("🎥 Starting camera feed... Please look directly at the webcam.")
                        roll_number, err = auto_scan_face_from_camera(timeout=25)
                        if roll_number:
                            success, msg = mark_attendance(roll_number, lab_choice)
                            if success:
                                play_siri_voice(True, roll_number)
                                st.balloons()
                                show_scan_popup(True, msg, roll_number)
                            else:
                                play_siri_voice(False, roll_number)
                                show_scan_popup(False, msg, roll_number)
                            st.rerun()
                        elif err:
                            st.error(f"Face verification failed: {err}")
                            
            else:
                st.subheader("📷 Automatic QR Code Scanner")
                st.write("Use your webcam for automatic QR Code scanning and attendance marking.")

                auto_scan_mode = st.checkbox(
                    "Enable automatic QR scanner",
                    value=st.session_state.auto_scan_active,
                    key="auto_scan_active",
                    help="Present your printed/digital QR code to the webcam to mark attendance."
                )

                if auto_scan_mode:
                    st.info("🎥 Scanning from your webcam. Present your QR Code clearly.")
                    roll_number, err = auto_scan_qr_from_camera(timeout=12)
                    if roll_number:
                        registry = load_student_registry()
                        registered_rolls = registry["Roll Number"].astype(str).tolist()
                        if not registered_rolls or roll_number in registered_rolls:
                            success, msg = mark_attendance(roll_number, lab_choice)
                        else:
                            success = False
                            msg = f"Student '{roll_number}' is not registered in the student database. Register them first."
                        
                        if success:
                            play_siri_voice(True, roll_number)
                            st.balloons()
                            show_scan_popup(True, msg, roll_number)
                        else:
                            play_siri_voice(False, roll_number)
                            show_scan_popup(False, msg, roll_number)
                    elif err:
                        st.error(f"QR Scanning failed: {err}")

                st.markdown("---", unsafe_allow_html=True)
                st.write("Manual upload / capture:")
                qr_file = st.file_uploader("Upload QR Code image:", type=["png", "jpg", "jpeg"], key="qr_file_uploader")
                st.write("--- or ---")
                st.write("Use Camera Capture:")
                qr_camera = st.camera_input("Capture QR snapshot:", key="qr_camera_input")
                
                scanned_file = qr_file or qr_camera
                if scanned_file is not None:
                    roll_number, err = decode_qr_code(scanned_file)
                    if roll_number:
                        registry = load_student_registry()
                        registered_rolls = registry["Roll Number"].astype(str).tolist()
                        if not registered_rolls or roll_number in registered_rolls:
                            success, msg = mark_attendance(roll_number, lab_choice)
                        else:
                            success = False
                            msg = f"Student '{roll_number}' is not registered in the student database. Register them first."
                        if success:
                            play_siri_voice(True, roll_number)
                            st.balloons()
                            show_scan_popup(True, msg, roll_number)
                        else:
                            play_siri_voice(False, roll_number)
                            show_scan_popup(False, msg, roll_number)
                    else:
                        st.error(f"QR Decoding failed: {err}")
                        
        with col2:
            st.subheader("Manual Attendance (Faculty Bypass)")
            if st.session_state.user_role == "student":
                roll_input = st.text_input("Your Roll Number:", value=st.session_state.username, disabled=True, key="manual_roll")
            else:
                roll_input = st.text_input("Enter Roll Number Manually:", key="manual_roll")
                
            if st.button("Submit Manual Entry", key="manual_mark"):
                roll_strip = roll_input.strip()
                if roll_strip:
                    registry = load_student_registry()
                    registered_rolls = registry["Roll Number"].astype(str).tolist()
                    
                    if not registered_rolls or roll_strip in registered_rolls:
                        success, msg = mark_attendance(roll_strip, lab_choice)
                    else:
                        success = False
                        msg = f"Roll Number '{roll_strip}' is not registered in the student database. Register them first."
                    
                    if success:
                        play_siri_voice(True, roll_strip)
                        st.balloons()
                        show_scan_popup(True, msg, roll_strip)
                    else:
                        play_siri_voice(False, roll_strip)
                        show_scan_popup(False, msg, roll_strip)
                else:
                    st.warning("Please enter a roll number.")
                    
        render_popup()
 
    if st.session_state.user_role == "student":
        with tabs[1]:
            st.header("👤 Register My Face")
            st.write("Capture or upload a photo of your face to register in the student database.")
            
            student_roll = st.text_input("Confirm Your Roll Number:", value=st.session_state.username if st.session_state.username != "Student" else "", key="student_roll_face_reg")
            
            st.write("Capture Face:")
            std_face_cam = st.camera_input("Capture snapshot", key="student_face_cam")
            st.write("--- or ---")
            std_face_upload = st.file_uploader("Upload photo of your face:", type=["png", "jpg", "jpeg"], key="student_face_upload")
            
            std_face_img = std_face_cam or std_face_upload
            
            if st.button("Register My Face", type="primary", key="student_register_face_btn"):
                roll_clean = student_roll.strip()
                if not roll_clean:
                    st.warning("Please confirm your Roll Number.")
                elif std_face_img is None:
                    st.warning("Please capture or upload a face picture.")
                else:
                    success, msg = enroll_student(roll_clean, std_face_img)
                    if success:
                        st.success(msg)
                        st.balloons()
                    else:
                        st.error(msg)
                        
        with tabs[2]:
            st.header("📇 Get My QR Code")
            st.write("Generate and download your personalized attendance QR Code.")
            
            student_roll = st.text_input("Enter your Roll Number:", placeholder="e.g. U16VH24S0208", key="student_roll_qr")
            if st.button("Generate My QR Code", type="primary", key="student_gen_qr_btn"):
                roll_clean = student_roll.strip()
                if roll_clean:
                    qr_img = generate_qr_code_image(roll_clean)
                    buf = BytesIO()
                    qr_img.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    st.image(byte_im, width=200, caption=f"QR Code for {roll_clean}")
                    st.download_button(
                        label="Download QR Code",
                        data=byte_im,
                        file_name=f"{roll_clean}_qrcode.png",
                        mime="image/png",
                        key=f"dl_student_{roll_clean}"
                    )
                else:
                    st.warning("Please enter a valid Roll Number.")
                    
            st.divider()
            st.subheader("How to use Face & QR Code Attendance")
            st.markdown("""
            - **Register Face**: Navigate to the `Register My Face` tab and capture/upload your face photo.
            - **Generate QR Code**: Download your personalized QR code as a secondary verification option.
            - **Mark Attendance**: Use either Face Recognition or QR Code Scanner to instantly verify your identity and mark attendance.
            """)
    else:
        with tabs[1]:
            st.header("Attendance Records")
            df = load_data()
        
        st.markdown("<div class='section-note'>Search and filter attendance entries with ease. Download the current view or remove outdated entries safely.</div>", unsafe_allow_html=True)
        
        # Date picker for filtering
        col_date, col_filter = st.columns([1, 1])
        with col_date:
            selected_date = st.date_input(
                "Select Date to View Records",
                value=datetime.now().date(),
                help="Choose a date to view attendance records for that specific day"
            )
        
        with col_filter:
            filter_lab = st.selectbox("Filter by Subject", ["All", "Python", "Operating System", "Computer Graphics", "DataStructure", "Cpp", "DBMS", "DigitalLogics", "JAVA", "WebDesigen", "C Programing", "DAA", "C#", "PHP", "AIML"])
        
        # Convert selected_date to string format for comparison
        selected_date_str = selected_date.strftime("%Y-%m-%d")
            
        # Filter data by selected date
        date_filtered_df = df[df["Date"] == selected_date_str]
        
        # Apply subject filter if needed
        if filter_lab != "All":
            display_df = date_filtered_df[date_filtered_df["Lab"] == filter_lab]
        else:
            display_df = date_filtered_df
        
        display_df = display_df.copy()
        display_df["Lab"] = display_df["Lab"].apply(lambda x: x + " (2nd sem)" if x in ["DataStructure", "Cpp"] else x)
        
        st.subheader(f"📅 Records for {selected_date.strftime('%B %d, %Y')} ({len(display_df)} total records)")
        
        # Download button for filtered data
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Filtered Records as CSV",
            csv,
            f"attendance_records_{selected_date_str}.csv",
            "text/csv",
            key='download-csv-filtered'
        )
        
        st.divider()
        
        # Display records organized by subject
        if not display_df.empty:
            # Group by subject and show each subject's records
            subjects = sorted(display_df["Lab"].unique())
            
            for subject in subjects:
                subject_records = display_df[display_df["Lab"] == subject].sort_values("Time", ascending=False)
                with st.expander(f"📚 {subject} ({len(subject_records)} students)", expanded=True):
                    st.dataframe(
                        subject_records[["Roll Number", "Time", "Lab"]], 
                        width='stretch', 
                        hide_index=True,
                        column_config={
                            "Roll Number": st.column_config.TextColumn("Student Roll Number", width="medium"),
                            "Time": st.column_config.TextColumn("Attendance Time", width="medium"),
                            "Lab": st.column_config.TextColumn("Subject", width="medium")
                        }
                    )
                    
                    # Summary stats for this subject
                    col_stats1, col_stats2 = st.columns(2)
                    with col_stats1:
                        st.metric("Total Students", len(subject_records))
                    with col_stats2:
                        # Check if it's Friday/Saturday for max attendance info
                        day_of_week = selected_date.weekday()
                        max_allowed = 2 if day_of_week in [4, 5] else 1
                        st.metric("Max Allowed per Student", max_allowed)
                    
                    st.subheader("Delete Record")
                    options = subject_records.apply(lambda row: f"{row['Roll Number']} - {row['Time']}", axis=1).tolist()
                    indices = subject_records.index.tolist()
                    option_to_index = dict(zip(options, indices))
                    
                    record_to_delete = st.selectbox(
                        f"Select record to delete for {subject}:", 
                        ["-- Select --"] + options, 
                        key=f"delete_select_{subject}_{selected_date_str}"
                    )
                    
                    if record_to_delete != "-- Select --":
                        if st.button("Delete Selected Record", type="primary", key=f"delete_btn_{subject}_{selected_date_str}"):
                            idx_to_drop = option_to_index[record_to_delete]
                            sel_rec = df.loc[idx_to_drop]
                            with get_db_connection() as conn:
                                conn.execute(
                                    "DELETE FROM attendance WHERE roll_number = ? AND date = ? AND time = ? AND lab = ?",
                                    (str(sel_rec["Roll Number"]), str(sel_rec["Date"]), str(sel_rec["Time"]), str(sel_rec["Lab"]))
                                )
                                conn.commit()
                            st.toast(f"✅ Deleted record for {record_to_delete.split(' - ')[0]}")
                            st.rerun()
            
            st.divider()
            st.subheader("Delete Records")
            st.write("Use these buttons to remove records for the selected date or the entire attendance history.")
            delete_date_button, delete_all_button = st.columns(2)
            with delete_date_button:
                if st.button(f"Delete all records for {selected_date.strftime('%b %d, %Y')}", key=f"delete_date_{selected_date_str}"):
                    with get_db_connection() as conn:
                        conn.execute("DELETE FROM attendance WHERE date = ?", (selected_date_str,))
                        conn.commit()
                    st.success(f"✅ Deleted all records for {selected_date.strftime('%b %d, %Y')}")
                    st.rerun()
            with delete_all_button:
                confirm_delete_all = st.checkbox(
                    "I understand this will permanently delete all attendance history.",
                    key="confirm_delete_all_history"
                )
                if confirm_delete_all:
                    delete_all_text = st.text_input(
                        "Type DELETE to confirm full attendance history deletion:",
                        key="confirm_delete_all_text"
                    )
                    if delete_all_text.strip().upper() == "DELETE":
                        if st.button("Delete all attendance history", key="delete_all_history"):
                            with get_db_connection() as conn:
                                conn.execute("DELETE FROM attendance")
                                conn.commit()
                            st.success("✅ Deleted all attendance records")
                            st.rerun()
                    else:
                        st.warning("Type DELETE exactly to enable the delete button.")
                else:
                    st.info("Check the box above to enable full history deletion.")
            
            # Overall summary for the selected date
            st.divider()
            st.subheader("📊 Daily Summary")
            col_sum1, col_sum2, col_sum3 = st.columns(3)
            
            total_students = len(display_df["Roll Number"].unique())
            total_records = len(display_df)
            day_name = selected_date.strftime("%A")
            
            with col_sum1:
                st.metric("Total Students", total_students)
            with col_sum2:
                st.metric("Total Records", total_records)
            with col_sum3:
                st.metric("Day", day_name)
                
        else:
            st.info(f"📭 No attendance records found for {selected_date.strftime('%B %d, %Y')}.")
            
            # Show available dates with records
            available_dates = sorted(df["Date"].unique(), reverse=True)
            if available_dates:
                st.write("**Available dates with records:**")
                date_options = [datetime.strptime(date, "%Y-%m-%d").strftime("%B %d, %Y (%A)") for date in available_dates[:10]]  # Show last 10 dates
                st.write(", ".join(date_options))
        
        # Option to view all records (original functionality)
        st.divider()
        with st.expander("📋 View All Records (Historical View)", expanded=False):
            st.markdown("### Complete Attendance History")
            
            filter_lab_all = st.selectbox("Filter by Subject (All Records)", ["All", "Python", "Operating System", "Computer Graphics", "DataStructure", "Cpp", "DBMS", "DigitalLogics", "JAVA", "WebDesigen", "C Programing", "R Programing", "C.prog", "MAD", "WCMS"], key="filter_all")
            
            if filter_lab_all != "All":
                original_display_df = df[df["Lab"] == filter_lab_all]
            else:
                original_display_df = df
                
            display_df_all = original_display_df.copy()
            display_df_all["Lab"] = display_df_all["Lab"].apply(lambda x: x + " (2nd sem)" if x in ["DataStructure", "Cpp"] else x)
            
            # Download button for all records
            csv_all = original_display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download All Records as CSV",
                csv_all,
                "all_attendance_records.csv",
                "text/csv",
                key='download-csv-all'
            )
            
            # Display all records organized by date
            if not display_df_all.empty:
                dates = sorted(display_df_all["Date"].unique(), reverse=True)
                for date in dates:
                    date_records = display_df_all[display_df_all["Date"] == date].sort_values("Time", ascending=False)
                    with st.expander(f"📅 {date} ({len(date_records)} records)", expanded=False):
                        st.dataframe(date_records[["Roll Number", "Time", "Lab"]], width='stretch', hide_index=True)
        
        with tabs[2]:
            st.header("🧑‍🎓 Student Face & QR Registry")
            st.markdown("<div class='section-note'>Register new student Roll Numbers along with their Face photo, list all registered students, and generate/download their personalized QR codes.</div>", unsafe_allow_html=True)
            
            st.subheader("Enroll New Student")
            enroll_roll = st.text_input("Enter Student Roll Number:", key="enroll_roll")
            
            fac_image = None
            if FACE_RECOGNITION_AVAILABLE:
                st.write("Student Face Photo (Required):")
                fac_cam = st.camera_input("Capture Student Face", key="fac_enroll_cam")
                st.write("--- or ---")
                fac_upload = st.file_uploader("Upload Student Face Photo:", type=["png", "jpg", "jpeg"], key="fac_enroll_upload")
                fac_image = fac_cam or fac_upload

            if st.button("Register Student & Save Face", type="primary", key="enroll_student_btn"):
                roll_clean = enroll_roll.strip()
                if not roll_clean:
                    st.warning("Please enter a valid Roll Number.")
                elif FACE_RECOGNITION_AVAILABLE and fac_image is None:
                    st.warning("Please capture or upload a face photo for the student.")
                else:
                    success, message = enroll_student(roll_clean, fac_image)
                    if success:
                        st.success(message)
                        st.balloons()
                        
                        # Generate and show QR Code
                        qr_img = generate_qr_code_image(roll_clean)
                        buf = BytesIO()
                        qr_img.save(buf, format="PNG")
                        byte_im = buf.getvalue()
                        
                        st.image(byte_im, width=200, caption=f"QR Code for {roll_clean}")
                        st.download_button(
                            label="Download QR Code",
                            data=byte_im,
                            file_name=f"{roll_clean}_qrcode.png",
                            mime="image/png",
                            key=f"dl_{roll_clean}"
                        )
                    else:
                        st.error(message)
            
            st.divider()
            st.subheader("Registered Students List")
            registry_df = load_student_registry()
            if not registry_df.empty:
                st.write(f"Total registered students: **{len(registry_df)}**")
                
                display_df = registry_df.copy()
                if "Face Encoding" in display_df.columns:
                    display_df["Face Registered"] = display_df["Face Encoding"].apply(
                        lambda x: "🟢 Registered" if isinstance(x, str) and len(x.strip()) > 10 else "🔴 Missing"
                    )
                    display_columns = ["Roll Number", "Registration Date", "Face Registered"]
                    if "Face Path" in display_df.columns:
                        display_columns.append("Face Path")
                    st.dataframe(display_df[display_columns], width='stretch', hide_index=True)
                else:
                    st.dataframe(display_df, width='stretch', hide_index=True)
                
                # Individual QR Code dropdown generator
                st.subheader("View/Download Existing Student QR Code")
                selected_student = st.selectbox("Select Student:", ["-- Select --"] + registry_df["Roll Number"].astype(str).tolist())
                if selected_student != "-- Select --":
                    qr_img = generate_qr_code_image(selected_student)
                    buf = BytesIO()
                    qr_img.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    st.image(byte_im, width=200, caption=f"QR Code for {selected_student}")
                    st.download_button(
                        label="Download QR Code",
                        data=byte_im,
                        file_name=f"{selected_student}_qrcode.png",
                        mime="image/png",
                        key=f"dl_existing_{selected_student}"
                    )
            else:
                st.info("No students registered yet. Enroll a student above to get started!")

            st.divider()
            st.subheader("Generate QR Code from Roll Number")
            generate_roll = st.text_input("Enter a roll number to generate a QR code", key="faculty_generate_qr_roll", placeholder="e.g. U16VH24S0208")
            if st.button("Generate QR Code", key="faculty_generate_qr_btn"):
                if generate_roll.strip():
                    generated_qr_img = generate_qr_code_image(generate_roll.strip())
                    buf = BytesIO()
                    generated_qr_img.save(buf, format="PNG")
                    generated_bytes = buf.getvalue()
                    st.image(generated_bytes, width=240, caption=f"QR Code for {generate_roll.strip()}")
                    st.download_button(
                        label="Download Generated QR Code",
                        data=generated_bytes,
                        file_name=f"{generate_roll.strip()}_qrcode.png",
                        mime="image/png",
                        key=f"faculty_download_qr_{generate_roll.strip()}"
                    )
                else:
                    st.warning("Please enter a roll number to generate a QR code.")

        with tabs[3]:
            st.header("📥 Download Student Attendance Record")
            st.markdown("<div class='section-note'>Enter a student's <strong>Roll Number</strong> to view and download their complete attendance history across <strong>all subjects</strong>.</div>", unsafe_allow_html=True)

            search_roll = st.text_input(
                "🔍 Enter Student Roll Number",
                placeholder="e.g. U16VH24S0208",
                key="faculty_search_student_roll"
            ).strip()

            if search_roll:
                all_data = load_data()
                student_data = all_data[all_data["Roll Number"].astype(str) == search_roll]

                if student_data.empty:
                    st.warning(f"⚠️ No attendance records found for Roll Number **{search_roll}**.")
                else:
                    # Summary metrics
                    total_records = len(student_data)
                    unique_subjects = student_data["Lab"].nunique()
                    unique_dates = student_data["Date"].nunique()

                    st.success(f"✅ Found **{total_records}** attendance records for **{search_roll}** across **{unique_subjects}** subjects over **{unique_dates}** days.")

                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Total Records", total_records)
                    with col_m2:
                        st.metric("Subjects", unique_subjects)
                    with col_m3:
                        st.metric("Days Attended", unique_dates)

                    st.divider()

                    # Download all records as CSV
                    csv_student = student_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"📥 Download All Records for {search_roll} as CSV",
                        data=csv_student,
                        file_name=f"{search_roll}_attendance_all_subjects.csv",
                        mime="text/csv",
                        key=f"dl_student_all_{search_roll}",
                        type="primary",
                        use_container_width=True
                    )

                    st.divider()

                    # Show records grouped by subject
                    st.subheader("📚 Subject-wise Attendance Breakdown")
                    subjects = sorted(student_data["Lab"].unique())

                    for subject in subjects:
                        subject_records = student_data[student_data["Lab"] == subject].sort_values("Date", ascending=False)
                        with st.expander(f"📘 {subject} — {len(subject_records)} record(s)", expanded=True):
                            st.dataframe(
                                subject_records[["Date", "Time", "Lab"]],
                                width='stretch',
                                hide_index=True,
                                column_config={
                                    "Date": st.column_config.TextColumn("Date", width="medium"),
                                    "Time": st.column_config.TextColumn("Time", width="medium"),
                                    "Lab": st.column_config.TextColumn("Subject", width="medium")
                                }
                            )

                            # Per-subject CSV download
                            csv_subject = subject_records.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label=f"📥 Download {subject} records",
                                data=csv_subject,
                                file_name=f"{search_roll}_{subject.replace(' ', '_')}_attendance.csv",
                                mime="text/csv",
                                key=f"dl_student_{search_roll}_{subject}"
                            )
            else:
                st.info("👆 Type a student Roll Number above to search their attendance records.")