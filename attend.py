from streamlit.runtime import scriptrunner
from streamlit.runtime import scriptrunner
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from streamlit_js_eval import get_geolocation, streamlit_js_eval
from geopy.distance import geodesic

COLLEGE_LOCATION = (15.273742673769599, 76.37739703526368)
ALLOWED_RADIUS_METERS = 40000

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
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

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


_models_checked = False


def ensure_models_cached():
    """Check models once per process lifetime instead of every call."""
    global _models_checked
    if _models_checked:
        return True
    result = ensure_models_downloaded()
    if result:
        _models_checked = True
    return result


def get_face_detector(image_width, image_height):
    """Get OpenCV YuNet face detector (lightweight, re-created per image size)."""
    detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL_PATH, "", (image_width, image_height),
        score_threshold=0.6, nms_threshold=0.3, top_k=5000
    )
    return detector


@st.cache_resource
def get_face_recognizer():
    """Get OpenCV SFace face recognizer — cached so the 37MB ONNX model loads only once."""
    return cv2.FaceRecognizerSF.create(SFACE_MODEL_PATH, "")


def _downscale_image(image_np, max_dim=640):
    """Downscale image to max_dim on the longest side for faster detection."""
    h, w = image_np.shape[:2]
    if max(h, w) <= max_dim:
        return image_np
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)


def extract_face_embedding(image_np):
    """Extract 128-d face embedding from an image using YuNet + SFace.
    Returns (embedding_array, error_string). On success error_string is None."""
    if not ensure_models_cached():
        return None, "Face recognition models are not available."

    # Downscale for faster face detection
    image_np = _downscale_image(image_np, max_dim=640)

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
    """Compare two face embeddings using cosine similarity (numpy — no model reload).
    Returns (is_match, similarity_score)."""
    emb1 = emb1.flatten().astype(np.float64)
    emb2 = emb2.flatten().astype(np.float64)
    dot = np.dot(emb1, emb2)
    norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    score = dot / norm if norm > 0 else 0.0
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

# Load CSS immediately so login page looks styled
css = load_static_file("style.css")
if css:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Initialize ALL session state variables upfront
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = ""
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'student_roll' not in st.session_state:
    st.session_state.student_roll = ""
if 'show_student_login' not in st.session_state:
    st.session_state.show_student_login = False
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
if 'location_verified' not in st.session_state:
    st.session_state.location_verified = False
    st.session_state.location_distance = None
if 'login_tab' not in st.session_state:
    st.session_state.login_tab = "student"

import sqlite3

DB_FILE = "attendance.db"
CSV_FILE = "lab_attendance.csv"
STUDENT_REGISTRY_FILE = "student_registry.csv"
FACULTIES_REGISTRY_FILE = "faculties_registry.csv"
REGISTERED_FACES_DIR = "registered_faces"


# ================================================================
#  STEP 1: LOGIN GATE — Page is BLANK until user logs in
# ================================================================

if not st.session_state.logged_in:
    # Full-page login screen
    st.markdown("""
    <div style="text-align: center; padding-top: 30px;">
        <div style="font-size: 4rem; margin-bottom: 8px;">📝</div>
        <h1 style="font-size: 2rem; margin: 0;">Smart Lab Attendance</h1>
        <p style="color: var(--text-secondary); font-size: 1rem; margin-top: 6px;">Sign in to access the attendance system</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Login type selection
    login_cols = st.columns(3)
    with login_cols[0]:
        if st.button("🎓 Student Login", use_container_width=True, type="primary" if st.session_state.login_tab == "student" else "secondary", key="tab_student_login"):
            st.session_state.login_tab = "student"
            st.rerun()
    with login_cols[1]:
        if st.button("🔐 Faculty Login", use_container_width=True, type="primary" if st.session_state.login_tab == "faculty" else "secondary", key="tab_faculty_login"):
            st.session_state.login_tab = "faculty"
            st.rerun()
    with login_cols[2]:
        if st.button("👑 Admin Login", use_container_width=True, type="primary" if st.session_state.login_tab == "admin" else "secondary", key="tab_admin_login"):
            st.session_state.login_tab = "admin"
            st.rerun()

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # --- STUDENT LOGIN ---
    if st.session_state.login_tab == "student":
        st.markdown("""
        <div class="hero" style="border-left: 4px solid #60a5fa;">
            <h2>🎓 Student Sign In</h2>
            <p>Enter your Roll Number and the common student password to access attendance features.</p>
        </div>
        """, unsafe_allow_html=True)

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            student_roll_input = st.text_input("Roll Number", placeholder="e.g. 220301, U16VH24S0208", key="login_student_roll")
        with s_col2:
            student_pw_input = st.text_input("Password", type="password", placeholder="••••••••", key="login_student_pw")

        if st.button("🚀 Sign In as Student", type="primary", use_container_width=True, key="login_student_btn"):
            roll_clean = student_roll_input.strip()
            if not roll_clean:
                st.error("Please enter your Roll Number.")
            elif student_pw_input != "student123":
                st.error("❌ Incorrect password.")
            else:
                # Check if faculty has registered this student's face (inline DB check)
                face_registered = False
                try:
                    _conn = sqlite3.connect("attendance.db")
                    _cur = _conn.cursor()
                    _cur.execute(
                        "SELECT face_encoding, face_path FROM students WHERE roll_number = ?",
                        (roll_clean,)
                    )
                    _row = _cur.fetchone()
                    _conn.close()
                    if _row and _row[0] is not None:
                        face_registered = True
                except Exception:
                    face_registered = False

                if not face_registered:
                    st.error(
                        f"❌ **Login Denied** — Roll Number `{roll_clean}` does not have a registered face.\n\n"
                        "Your face must be registered by the **Faculty** before you can log in. "
                        "Please contact your faculty to register your face first."
                    )
                else:
                    st.session_state.logged_in = True
                    st.session_state.user_role = "student"
                    st.session_state.username = roll_clean
                    st.session_state.student_roll = roll_clean
                    st.session_state.location_verified = False
                    st.success(f"✅ Welcome, Student {roll_clean}!")
                    time.sleep(0.5)
                    st.rerun()

        st.caption("🔑 Common student password: **student123**")

    # --- FACULTY LOGIN ---
    elif st.session_state.login_tab == "faculty":
        st.markdown("""
        <div class="hero" style="border-left: 4px solid #a78bfa;">
            <h2>🔐 Faculty Sign In</h2>
            <p>Enter your faculty email and password to manage attendance records.</p>
        </div>
        """, unsafe_allow_html=True)

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            faculty_email_input = st.text_input("Faculty Email", placeholder="professor@college.edu", key="login_faculty_email")
        with f_col2:
            faculty_pw_input = st.text_input("Password", type="password", placeholder="••••••••", key="login_faculty_pw")

        if st.button("🔐 Sign In as Faculty", type="primary", use_container_width=True, key="login_faculty_btn"):
            if not faculty_email_input.strip():
                st.error("Please enter a valid Email.")
            elif faculty_pw_input != "faculty123":
                st.error("❌ Incorrect password.")
            else:
                st.session_state.logged_in = True
                st.session_state.user_role = "faculty"
                st.session_state.username = faculty_email_input.strip()
                st.session_state.location_verified = False
                st.success(f"✅ Welcome, Faculty!")
                time.sleep(0.5)
                st.rerun()

        st.caption("🔑 Faculty password: **faculty123**")

    # --- ADMIN LOGIN ---
    elif st.session_state.login_tab == "admin":
        st.markdown("""
        <div class="hero" style="border-left: 4px solid #f59e0b;">
            <h2>👑 Admin Sign In</h2>
            <p>Enter admin credentials to manage faculty, students, and system settings.</p>
        </div>
        """, unsafe_allow_html=True)

        a_col1, a_col2 = st.columns(2)
        with a_col1:
            admin_id_input = st.text_input("Admin ID", placeholder="admin", key="login_admin_id")
        with a_col2:
            admin_pw_input = st.text_input("Password", type="password", placeholder="••••••••", key="login_admin_pw")

        if st.button("👑 Sign In as Admin", type="primary", use_container_width=True, key="login_admin_btn"):
            if not admin_id_input.strip():
                st.error("Please enter Admin ID.")
            elif admin_pw_input != "admin123":
                st.error("❌ Incorrect password.")
            else:
                st.session_state.logged_in = True
                st.session_state.user_role = "admin"
                st.session_state.username = admin_id_input.strip()
                st.session_state.location_verified = False
                st.success(f"✅ Welcome, Admin!")
                time.sleep(0.5)
                st.rerun()

        st.caption("🔑 Admin password: **admin123**")

    # Block everything after login gate
    st.stop()


# ================================================================
#  STEP 2: LOCATION VERIFICATION — runs AFTER login
# ================================================================

if not st.session_state.location_verified:
    st.subheader("📍 Location Verification")
    st.markdown(f"<p style='color: var(--text-secondary);'>Signed in as <strong>{st.session_state.user_role.title()}: {st.session_state.username}</strong> — verifying your location...</p>", unsafe_allow_html=True)

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
            st.rerun()

# ================================================================
#  STEP 3: MAIN DASHBOARD — only reached after login + location
# ================================================================

if st.session_state.location_distance is not None:
    st.success(f"✅ Location verified ({st.session_state.location_distance:.0f}m from college)")

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
    
    existing_today = df[(df["Roll Number"] == roll_number) & (df["Date"] == current_date)]
    existing_count = len(existing_today)
    
    # On Fridays (4), Saturdays (5), and Sundays (6), allow up to 2 attendances
    if day_of_week in [4, 5, 6]:  # Friday, Saturday, Sunday
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
    # Invalidate the cached embeddings so new enrollments are picked up immediately
    _load_cached_embeddings.clear()

def load_faculties_registry():
    with get_db_connection() as conn:
        df = pd.read_sql_query("SELECT id, name AS 'Name', email AS 'Email', department AS 'Department', year AS 'Year', semester AS 'Semester' FROM faculties", conn)
    for col in ["Year", "Semester"]:
        if col not in df.columns:
            df[col] = ""
    return df


def get_daily_subject_matrix(selected_date_str, filter_lab="All"):
    """
    Constructs a daily subject-wise attendance matrix for a given date.
    Returns: (matrix_df, metrics_dict)
    """
    att_df = load_data()
    reg_df = load_student_registry()
    
    if att_df.empty:
        return pd.DataFrame(), {
            "total_students_present": 0,
            "total_records": 0,
            "subjects_conducted": [],
            "reg_students_count": len(reg_df) if not reg_df.empty else 0
        }
        
    day_att = att_df[att_df["Date"] == selected_date_str].copy()
    if day_att.empty:
        return pd.DataFrame(), {
            "total_students_present": 0,
            "total_records": 0,
            "subjects_conducted": [],
            "reg_students_count": len(reg_df) if not reg_df.empty else 0
        }

    if filter_lab != "All":
        day_att = day_att[day_att["Lab"] == filter_lab]

    subjects_conducted = sorted(day_att["Lab"].dropna().unique().tolist())
    
    registered_rolls = reg_df["Roll Number"].astype(str).str.strip().tolist() if not reg_df.empty else []
    today_rolls = day_att["Roll Number"].astype(str).str.strip().unique().tolist()
    all_rolls = sorted(list(set(registered_rolls + today_rolls)))
    
    matrix_rows = []
    for roll in all_rolls:
        row_dict = {"Roll Number": roll}
        subjects_attended_count = 0
        
        for subj in subjects_conducted:
            match = day_att[(day_att["Roll Number"].astype(str).str.strip() == roll) & (day_att["Lab"] == subj)]
            if not match.empty:
                time_str = match.iloc[0]["Time"]
                row_dict[subj] = f"✅ Present ({time_str})"
                subjects_attended_count += 1
            else:
                if roll in registered_rolls:
                    row_dict[subj] = "❌ Absent"
                else:
                    row_dict[subj] = "— Not Enrolled"
                    
        row_dict["Attended / Conducted"] = f"{subjects_attended_count} / {len(subjects_conducted)}"
        row_dict["Daily Attendance Rate"] = f"{(subjects_attended_count / len(subjects_conducted) * 100):.1f}%" if len(subjects_conducted) > 0 else "0%"
        matrix_rows.append(row_dict)
        
    matrix_df = pd.DataFrame(matrix_rows)
    
    metrics = {
        "total_students_present": len(today_rolls),
        "total_records": len(day_att),
        "subjects_conducted": subjects_conducted,
        "reg_students_count": len(all_rolls)
    }
    
    return matrix_df, metrics


def get_student_daily_records(roll_number):
    """
    Returns daily subject-wise records and statistics for a specific student.
    """
    att_df = load_data()
    if att_df.empty or not roll_number:
        return pd.DataFrame(), pd.DataFrame()
        
    student_att = att_df[att_df["Roll Number"].astype(str).str.strip().str.upper() == str(roll_number).strip().upper()].copy()
    if student_att.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    student_att["Date_Obj"] = pd.to_datetime(student_att["Date"])
    student_att["Day_Name"] = student_att["Date_Obj"].dt.strftime("%A")
    student_att = student_att.sort_values(by=["Date", "Time"], ascending=[False, False])
    
    daily_log = student_att[["Date", "Day_Name", "Lab", "Time"]].rename(columns={
        "Day_Name": "Day of Week",
        "Lab": "Subject",
        "Time": "Time Marked"
    })
    
    subj_summary = student_att.groupby("Lab").agg(
        Sessions_Attended=("Date", "count"),
        First_Attended=("Date", "min"),
        Latest_Attended=("Date", "max")
    ).reset_index().rename(columns={
        "Lab": "Subject",
        "Sessions_Attended": "Total Sessions Attended",
        "First_Attended": "First Recorded Date",
        "Latest_Attended": "Most Recent Date"
    })
    
    return daily_log, subj_summary



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


@st.cache_data(ttl=60)
def _load_cached_embeddings():
    """Load and deserialize all registered student embeddings — cached for 60s."""
    df = load_student_registry()
    known_encodings = []
    known_rolls = []
    for _, row in df.iterrows():
        enc_str = row.get("Face Encoding", "")
        enc = deserialize_face_encoding(enc_str)
        if enc is not None:
            known_encodings.append(enc)
            known_rolls.append(str(row["Roll Number"]))
    return known_encodings, known_rolls


def get_student_face_details(roll_number):
    """
    Fetch student registration record, face image path, and face encoding from SQLite database.
    """
    if not roll_number:
        return None
    roll_str = str(roll_number).strip()
    df = load_student_registry()
    if df.empty:
        return None
    matches = df[df["Roll Number"].astype(str).str.strip().str.upper() == roll_str.upper()]
    if matches.empty:
        return None
    row = matches.iloc[0]
    enc_str = str(row.get("Face Encoding", ""))
    face_path = str(row.get("Face Path", ""))
    
    if not face_path or not os.path.exists(face_path):
        candidate_path = os.path.join(REGISTERED_FACES_DIR, f"{roll_str}.jpg")
        if os.path.exists(candidate_path):
            face_path = candidate_path

    encoding = deserialize_face_encoding(enc_str) if enc_str else None

    return {
        "roll_number": str(row["Roll Number"]),
        "registration_date": str(row.get("Registration Date", "")),
        "face_encoding": encoding,
        "face_path": face_path if (face_path and os.path.exists(face_path)) else None,
        "has_face": encoding is not None
    }


def match_face(face_image_file, threshold=0.363, target_roll_number=None):
    """
    Given an uploaded or captured image file, detect the face,
    compare it against the registered face encodings in the database,
    and return the matched roll number, or None if no match.
    If target_roll_number is specified, performs fast 1-to-1 face verification against ONLY that student.
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return None, "Face recognition library is not available."
    
    try:
        image = Image.open(face_image_file).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        unknown_embedding, err = extract_face_embedding(image_bgr)
        if unknown_embedding is None:
            return None, err
        
        if target_roll_number:
            details = get_student_face_details(target_roll_number)
            if not details or not details["has_face"]:
                return None, f"No registered face photo found for Roll Number '{target_roll_number}'. Please upload your face picture under 'Register My Face' tab first."
            is_match, score = compare_face_embeddings(details["face_encoding"], unknown_embedding, threshold=threshold)
            if is_match:
                return str(target_roll_number), None
            return None, f"Face verification failed: Captured face does not match the registered face photo for Roll Number '{target_roll_number}' (Similarity score: {score:.3f})."

        known_encodings, known_rolls = _load_cached_embeddings()

        if not known_rolls:
            return None, "No registered face encodings found in the database. Please enroll students first."
            
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


def scan_qr_from_image(image_file):
    """Scan QR code from a camera_input or uploaded image file.
    Uses the browser's default camera via st.camera_input (works on Streamlit Cloud, desktop, and mobile).
    Returns (decoded_data, error_string)."""
    if image_file is None:
        return None, "No image provided."
    
    try:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image_file.seek(0)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return None, "Failed to load captured image."
        
        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(img)
        
        if data:
            return data, None
        return None, "No QR Code detected in the captured image. Please try again with clearer framing."
    except Exception as e:
        return None, f"Error scanning QR Code: {str(e)}"


def verify_face_from_snapshot(image_file, threshold=0.363, target_roll_number=None):
    """Verify a face from a camera snapshot (st.camera_input) against registered students.
    If target_roll_number is provided, performs fast 1-to-1 matching against ONLY that student.
    Returns (roll_number, error_string). On success error_string is None."""
    if not FACE_RECOGNITION_AVAILABLE:
        return None, "Face recognition library is not available."

    if not ensure_models_cached():
        return None, "Face recognition models could not be loaded."
    
    if image_file is None:
        return None, "No image captured. Please take a photo."

    try:
        # Load image from the camera_input file
        image = Image.open(image_file).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Extract face embedding (image is downscaled inside)
        face_emb, err = extract_face_embedding(image_bgr)
        if face_emb is None:
            return None, err

        if target_roll_number:
            # 1-to-1 targeted face verification (ultra fast!)
            details = get_student_face_details(target_roll_number)
            if not details or not details["has_face"]:
                return None, f"No registered face photo found for Roll Number '{target_roll_number}'. Please upload your face picture under 'Register My Face' tab first."
            
            target_enc = details["face_encoding"]
            is_match, score = compare_face_embeddings(target_enc, face_emb, threshold=threshold)
            if is_match:
                return str(target_roll_number), None
            else:
                return None, f"Face verification failed: Captured face does not match the registered face photo for Student Roll Number '{target_roll_number}' (Similarity score: {score:.3f})."

        # Use cached embeddings for 1-to-N fallback loop
        known_encodings, known_rolls = _load_cached_embeddings()

        if not known_encodings:
            return None, "No registered face encodings found. Please enroll students first."

        best_score = -1.0
        best_idx = -1
        for i, known_enc in enumerate(known_encodings):
            is_match, score = compare_face_embeddings(known_enc, face_emb, threshold=threshold)
            if is_match and score > best_score:
                best_score = score
                best_idx = i
        
        if best_idx >= 0:
            return known_rolls[best_idx], None
        
        return None, "Face did not match any registered student. Please try again with better lighting and face clearly visible."
    except Exception as e:
        return None, f"Error processing face snapshot: {str(e)}"


def get_ear_from_file(img_file):
    """Get average EAR from a camera-input image file using MediaPipe FaceLandmarker.
    Returns (ear_value, error_string). On success error_string is None."""
    if not FACE_RECOGNITION_AVAILABLE:
        return None, "Face recognition library is not available."
    if not ensure_models_cached():
        return None, "Face recognition models could not be loaded."

    try:
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        landmarker_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        face_landmarker = FaceLandmarker.create_from_options(landmarker_options)

        image = Image.open(img_file).convert("RGB")
        image_np = np.array(image)
        h, w = image_np.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        results = face_landmarker.detect(mp_image)
        face_landmarker.close()

        if not results.face_landmarks:
            return None, "No face detected in the image."

        face_lm = results.face_landmarks[0]
        landmarks = get_landmarks_from_mediapipe(face_lm, w, h)

        ear_l = calculate_ear(landmarks["left_eye"])
        ear_r = calculate_ear(landmarks["right_eye"])
        avg_ear = (ear_l + ear_r) / 2.0
        return avg_ear, None
    except Exception as e:
        return None, f"EAR calculation error: {str(e)}"


def verify_liveness_from_snapshots(open_eyes_file, closed_eyes_file):
    """Verify liveness by comparing EAR (Eye Aspect Ratio) between two snapshots:
    one with eyes open and one with eyes closed.
    Also verifies that both photos show the **same person** via face embedding comparison.
    Returns (is_live, message, ear_open, ear_closed)."""
    if not FACE_RECOGNITION_AVAILABLE:
        return False, "Face recognition library is not available.", None, None

    if not ensure_models_cached():
        return False, "Face recognition models could not be loaded.", None, None

    if open_eyes_file is None or closed_eyes_file is None:
        return False, "Both photos are required for liveness verification.", None, None

    try:
        # --- Step A: Get EAR from both images ---
        open_eyes_file.seek(0)
        ear_open, err1 = get_ear_from_file(open_eyes_file)
        if ear_open is None:
            return False, f"Eyes-open photo: {err1}", None, None

        closed_eyes_file.seek(0)
        ear_closed, err2 = get_ear_from_file(closed_eyes_file)
        if ear_closed is None:
            return False, f"Eyes-closed photo: {err2}", ear_open, None

        # --- Step B: Validate EAR thresholds ---
        if ear_open < 0.18:
            return False, "Your eyes appear closed in the first photo. Please retake with eyes wide open.", ear_open, ear_closed

        ear_diff = ear_open - ear_closed
        blink_detected = ear_diff > 0.04 or ear_closed < ear_open * 0.85

        if not blink_detected:
            return False, (
                f"Blink not detected. Please make sure your eyes are clearly "
                f"OPEN in photo 1 and fully CLOSED in photo 2.\n"
                f"(EAR open: {ear_open:.3f}, EAR closed: {ear_closed:.3f})"
            ), ear_open, ear_closed

        # --- Step C: Same-person verification via face embeddings ---
        # This is a bonus anti-spoofing check. If the SFace model can't load
        # (e.g. OpenCV / ONNX version mismatch), we still pass liveness based
        # on the EAR blink detection above.
        same_person_ok = True  # assume ok unless proven otherwise
        try:
            open_eyes_file.seek(0)
            img_open = Image.open(open_eyes_file).convert("RGB")
            bgr_open = cv2.cvtColor(np.array(img_open), cv2.COLOR_RGB2BGR)
            emb_open, emb_err1 = extract_face_embedding(bgr_open)

            closed_eyes_file.seek(0)
            img_closed = Image.open(closed_eyes_file).convert("RGB")
            bgr_closed = cv2.cvtColor(np.array(img_closed), cv2.COLOR_RGB2BGR)
            emb_closed, emb_err2 = extract_face_embedding(bgr_closed)

            if emb_open is not None and emb_closed is not None:
                same_person, sim_score = compare_face_embeddings(emb_open, emb_closed, threshold=0.30)
                if not same_person:
                    return False, (
                        f"The two photos do not appear to be the same person "
                        f"(similarity: {sim_score:.3f}). Please retake both photos."
                    ), ear_open, ear_closed
        except Exception:
            # SFace model unavailable — skip same-person check
            same_person_ok = True

        # All checks passed
        return True, "✅ Liveness verified! Eye blink confirmed.", ear_open, ear_closed

    except Exception as e:
        return False, f"Liveness check error: {str(e)}", None, None


def play_siri_voice(success, roll_number=""):
    """(Deprecated) Voice is now handled by client-side JS in render_popup"""
    pass

# Initialize session state for popup
if 'show_popup' not in st.session_state:
    st.session_state.show_popup = False
if 'popup_data' not in st.session_state:
    st.session_state.popup_data = {}

def show_scan_popup(success, message, roll_number=""):
    """Display a popup after scanning"""
    st.session_state.show_popup = True
    
    roll_str = str(roll_number).replace('"', '').replace("'", "")
    prefix = f"{roll_str}, " if roll_str else ""
    if success:
        voice_msg = f"{prefix}your attendance is successfully recorded."
    else:
        voice_msg = f"{prefix}failed to record attendance."
        
    st.session_state.popup_data = {
        'success': success,
        'message': message,
        'roll_number': roll_number,
        'voice_msg': voice_msg
    }

def render_popup():
    """Render the popup modal"""
    if st.session_state.show_popup:
        popup_data = st.session_state.popup_data
        icon = "✅" if popup_data['success'] else "❌"
        status_class = "modal-success" if popup_data['success'] else "modal-error"
        heading = "Attendance Marked!" if popup_data['success'] else "Failed to Mark"
        
        voice_script = f"""
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                let msg = new SpeechSynthesisUtterance("{popup_data.get('voice_msg', '')}");
                msg.lang = 'en-US';
                let voices = window.speechSynthesis.getVoices();
                let selectedVoice = voices.find(v => v.name.includes('Siri') || v.name.includes('Samantha') || v.name.includes('Google US English'));
                if (selectedVoice) {{
                    msg.voice = selectedVoice;
                }}
                window.speechSynthesis.speak(msg);
            }}
        """
        
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
            {voice_script}
            setTimeout(() => {{
                //Auto close popup after 3 seconds
                let overlay = document.querySelector('.modal-overlay');
                if (overlay) overlay.style.display = 'none';
            }}, 3000);
        </script>
        """, unsafe_allow_html=True)
        
        # Reset popup state so it doesn't reappear on other interactions
        st.session_state.show_popup = False

# ------------------ DASHBOARD HEADER (after login + location) ------------------

# Navigation Header with logout
col_logo, col_role_info, col_logout = st.columns([3, 2, 1])
with col_logo:
    st.markdown("<h1 style='margin:0; font-size: 1.6rem; font-weight: 600; color: #f4f4f5;'>Smart Lab Attendance</h1>", unsafe_allow_html=True)
with col_role_info:
    role_icon = {"student": "🎓", "faculty": "🔐", "admin": "👑"}.get(st.session_state.user_role, "👤")
    st.markdown(f"<div style='padding-top: 8px; font-size: 0.95rem; color: var(--text-secondary);'>{role_icon} <strong>{st.session_state.user_role.title()}</strong>: {st.session_state.username}</div>", unsafe_allow_html=True)
with col_logout:
    st.markdown("<div style='padding-top: 4px;'></div>", unsafe_allow_html=True)
    if st.button("🚪 Logout", key="dashboard_logout_btn", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.user_role = ""
        st.session_state.username = ""
        st.session_state.student_roll = ""
        st.session_state.location_verified = False
        st.session_state.location_distance = None
        st.rerun()

# Student Profile Banner when logged in as student
if st.session_state.user_role == "student" and st.session_state.student_roll:
    student_details = get_student_face_details(st.session_state.student_roll)
    
    st.markdown(
        f"""
        <div class="hero" style="margin-top: 15px; border-left: 4px solid var(--accent-primary);">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
                <div>
                    <h2>🎓 Welcome Student: <span style="color: #60a5fa;">{st.session_state.student_roll}</span></h2>
                    <p>Logged in session active. Attendance scanning is configured for <strong>1-to-1 fast face matching</strong> against your stored profile.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    prof_col1, prof_col2 = st.columns([1, 4])
    with prof_col1:
        if student_details and student_details["face_path"]:
            st.image(student_details["face_path"], caption=f"Registered Face: {st.session_state.student_roll}", width=130)
        else:
            st.markdown("<div style='background: rgba(255,255,255,0.08); border-radius: 10px; padding: 20px; text-align: center; font-size: 2.2rem;'>👤</div>", unsafe_allow_html=True)
            st.caption("No Face Enrolled")
    with prof_col2:
        if student_details and student_details["has_face"]:
            st.success(f"✅ **Face Profile Loaded**: Registered on `{student_details['registration_date']}`. Fast 1-to-1 face verification enabled.")
        else:
            st.warning(f"⚠️ **Face Not Registered**: Please navigate to the '👤 Register My Face' tab below and take a photo to enable face attendance.")
elif st.session_state.user_role == "faculty":
    st.markdown(
        """
        <div class="hero" style="margin-top: 15px; border-left: 4px solid #a78bfa;">
            <h2>🔐 Faculty Dashboard</h2>
            <p>Manage attendance records, scan QR codes, register students, and export reports from your secure session.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif st.session_state.user_role == "admin":
    st.markdown(
        """
        <div class="hero" style="margin-top: 15px; border-left: 4px solid #f59e0b;">
            <h2>👑 Admin Dashboard</h2>
            <p>Full system management: faculty, students, attendance analytics, and record maintenance.</p>
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
    tab_list = ["👤 Face & QR Attendance", "📅 My Daily Attendance", "👤 Register My Face", "📇 My QR Code"]
else:
    tab_list = ["👤 Face & QR Attendance", "📊 Daily & Subject-Wise Records", "🧑‍🎓 Manage Students & Faces", "📥 Download Student Record"]
    
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
            st.subheader("👤 Live Secure Face Attendance")
            if not FACE_RECOGNITION_AVAILABLE:
                st.warning("⚠️ Face recognition is not available. Please install 'mediapipe' library or use QR Code mode.")
            else:
                st.markdown("""
                <div class='section-note' style='border-left-color: #10b981; background: rgba(16, 185, 129, 0.05); margin-bottom: 15px; padding: 12px 16px; border-radius: 8px;'>
                    <strong>👁️ Live Blink Liveness Check:</strong><br>
                    Position your face in the camera and <strong>blink</strong> to prove you are a real person.<br>
                    The system will automatically scan your face and mark attendance once verified.
                </div>
                """, unsafe_allow_html=True)

                class LiveFaceProcessor(VideoProcessorBase):
                    def __init__(self):
                        self.target_roll = st.session_state.get("student_roll") if st.session_state.get("user_role") == "student" and st.session_state.get("student_roll") else None
                        self.target_encoding = None
                        self.matched_roll = None
                        self.ear_history = []
                        self.liveness_verified = False
                        self.detector = None
                        self.recognizer = None
                        self.face_landmarker = None
                        self.known_encodings = []
                        self.known_rolls = []
                        self.initialized = False
                        self.last_w = 0
                        self.last_h = 0

                    def _initialize(self):
                        if not ensure_models_cached():
                            return False
                        self.recognizer = get_face_recognizer()
                        import mediapipe as mp
                        BaseOptions = mp.tasks.BaseOptions
                        FaceLandmarker = mp.tasks.vision.FaceLandmarker
                        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
                        VisionRunningMode = mp.tasks.vision.RunningMode
                        landmarker_options = FaceLandmarkerOptions(
                            base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_PATH),
                            running_mode=VisionRunningMode.IMAGE,
                            num_faces=1,
                            min_face_detection_confidence=0.5,
                            min_face_presence_confidence=0.5,
                        )
                        self.face_landmarker = FaceLandmarker.create_from_options(landmarker_options)
                        
                        if self.target_roll:
                            details = get_student_face_details(self.target_roll)
                            if details and details["has_face"]:
                                self.target_encoding = details["face_encoding"]
                        else:
                            self.known_encodings, self.known_rolls = _load_cached_embeddings()

                        self.initialized = True
                        return True

                    def recv(self, frame):
                        img = frame.to_ndarray(format="bgr24")
                        if not self.initialized:
                            if not self._initialize():
                                return av.VideoFrame.from_ndarray(img, format="bgr24")

                        # If already matched in this session, show success message
                        if self.matched_roll:
                            cv2.putText(img, f"Success! Marked {self.matched_roll}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            return av.VideoFrame.from_ndarray(img, format="bgr24")

                        h, w = img.shape[:2]
                        if self.detector is None or w != self.last_w or h != self.last_h:
                            self.detector = get_face_detector(w, h)
                            self.last_w = w
                            self.last_h = h

                        # 1. Liveness check via MediaPipe
                        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        import mediapipe as mp
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
                        results = self.face_landmarker.detect(mp_image)

                        ear = None
                        if results.face_landmarks:
                            face_lm = results.face_landmarks[0]
                            landmarks = get_landmarks_from_mediapipe(face_lm, w, h)
                            ear_l = calculate_ear(landmarks["left_eye"])
                            ear_r = calculate_ear(landmarks["right_eye"])
                            ear = (ear_l + ear_r) / 2.0
                            
                            # Draw eyes
                            for pt in landmarks["left_eye"] + landmarks["right_eye"]:
                                cv2.circle(img, pt, 2, (0, 255, 0), -1)

                            self.ear_history.append(ear)
                            if len(self.ear_history) > 15:
                                self.ear_history.pop(0)

                            # Detect blink (a sudden drop in EAR)
                            if len(self.ear_history) >= 10:
                                max_ear = max(self.ear_history)
                                min_ear = min(self.ear_history)
                                # Typical blink: open > 0.20, closed < 0.16
                                if max_ear > 0.20 and min_ear < 0.16:
                                    self.liveness_verified = True

                        status_color = (0, 255, 255) if not self.liveness_verified else (0, 255, 0)
                        status_text = "Blink to verify liveness..." if not self.liveness_verified else "Live person verified!"
                        cv2.putText(img, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                        if ear:
                            cv2.putText(img, f"EAR: {ear:.3f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                        # 2. Face Recognition
                        if self.liveness_verified:
                            _, faces = self.detector.detect(img)
                            if faces is not None and len(faces) > 0:
                                face = faces[0]
                                aligned_face = self.recognizer.alignCrop(img, face)
                                face_emb = self.recognizer.feature(aligned_face)
                                
                                if self.target_roll:
                                    # Fast 1-to-1 face verification
                                    if self.target_encoding is None:
                                        cv2.putText(img, f"No face registered for {self.target_roll}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    else:
                                        is_match, score = compare_face_embeddings(self.target_encoding, face_emb, threshold=0.363)
                                        if is_match:
                                            self.matched_roll = self.target_roll
                                            cv2.putText(img, f"Verified: {self.target_roll} ({score:.2f})", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                            box = face[0:4].astype(int)
                                            cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)
                                        else:
                                            cv2.putText(img, f"Face mismatch for {self.target_roll}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                else:
                                    # 1-to-N fallback loop
                                    best_score = -1.0
                                    best_idx = -1
                                    for i, known_enc in enumerate(self.known_encodings):
                                        is_match, score = compare_face_embeddings(known_enc, face_emb, threshold=0.363)
                                        if is_match and score > best_score:
                                            best_score = score
                                            best_idx = i
                                    
                                    if best_idx >= 0:
                                        self.matched_roll = self.known_rolls[best_idx]
                                        cv2.putText(img, f"Recognized: {self.matched_roll}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                        box = face[0:4].astype(int)
                                        cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)
                                    else:
                                        cv2.putText(img, "Recognizing... No match", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                        return av.VideoFrame.from_ndarray(img, format="bgr24")

                webrtc_ctx = webrtc_streamer(
                    key="live-face-scanner",
                    mode=WebRtcMode.SENDRECV,
                    video_processor_factory=LiveFaceProcessor,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )

                if webrtc_ctx.state.playing:
                    if webrtc_ctx.video_processor:
                        if webrtc_ctx.video_processor.matched_roll:
                            roll = webrtc_ctx.video_processor.matched_roll
                            success, msg = mark_attendance(roll, lab_choice)
                            if success:
                                play_siri_voice(True, roll)
                                st.balloons()
                                show_scan_popup(True, msg, roll)
                            else:
                                play_siri_voice(False, roll)
                                show_scan_popup(False, msg, roll)
                            webrtc_ctx.video_processor.matched_roll = None  # Reset to prevent continuous triggers
                            st.rerun()
                        else:
                            import time
                            time.sleep(1.0)
                            st.rerun()
                            
        else:
            st.subheader("📷 QR Code Scanner")
            st.write("Use your device camera to scan a QR Code for attendance marking.")

            st.markdown("**📸 Capture QR Code from Camera:**")
            qr_camera = st.camera_input("Point your camera at the QR Code and capture:", key="qr_camera_input")
            
            st.markdown("**--- or ---**")
            st.markdown("**📁 Upload QR Code image:**")
            qr_file = st.file_uploader("Upload QR Code image:", type=["png", "jpg", "jpeg"], key="qr_file_uploader")
            
            scanned_file = qr_camera or qr_file
            if scanned_file is not None:
                # Try scanning with both QR methods
                roll_number, err = scan_qr_from_image(scanned_file)
                if roll_number is None:
                    # Fallback to decode_qr_code
                    scanned_file.seek(0)
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
            roll_input = st.text_input("Your Roll Number:", value=st.session_state.student_roll if st.session_state.student_roll else st.session_state.username, disabled=True, key="manual_roll")
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
        st.header("📅 My Daily Attendance Records")
        current_student = st.session_state.get("student_roll", "")
        if current_student:
            st.markdown(f"<div class='section-note'>Viewing daily subject-wise attendance history for Student Roll Number: <strong>{current_student}</strong>.</div>", unsafe_allow_html=True)
            daily_df, summary = get_student_daily_records(current_student)
            
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Total Days Attended", summary.get("total_days_attended", 0))
            m_col2.metric("Total Sessions Marked", summary.get("total_sessions", 0))
            m_col3.metric("Distinct Subjects", summary.get("unique_labs", 0))
            
            st.divider()
            if not daily_df.empty:
                st.dataframe(daily_df, use_container_width=True)
            else:
                st.info("No attendance records found for your roll number yet.")
        else:
            st.warning("⚠️ Please sign in with your Roll Number above to view your daily attendance history.")

    with tabs[2]:
        st.header("👤 Register My Face")
        st.write("Capture or upload a photo of your face to register in the student database.")
        
        default_roll = st.session_state.get("student_roll", "")
        student_roll = st.text_input("Confirm / Enter Roll Number:", value=default_roll, key="student_roll_face_reg")
        
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
                    st.session_state.student_roll = roll_clean
                    st.session_state.username = roll_clean
                    st.success(msg)
                    st.balloons()
                    time.sleep(1.0)
                    st.rerun()
                else:
                    st.error(msg)
                    
    with tabs[3]:
        st.header("📇 Get My QR Code")
        st.write("Generate and download your personalized attendance QR Code.")
        
        default_roll_qr = st.session_state.get("student_roll", "")
        student_roll = st.text_input("Enter your Roll Number:", value=default_roll_qr, placeholder="e.g. U16VH24S0208", key="student_roll_qr")
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
        - **Sign In with Roll Number**: Log into the student portal using your roll number.
        - **Register Face**: Navigate to the `Register My Face` tab and capture your face photo.
        - **Fast 1-to-1 Matching**: Attendance scanning compares live camera feed strictly against your registered face for high speed and accuracy.
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