import streamlit as st
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

# Face recognition check is no longer needed since we use QR code scanning.
FACE_RECOGNITION_AVAILABLE = False

st.set_page_config(page_title="Smart Lab Attendance", layout="wide", page_icon="📝")

def load_static_file(filename: str) -> str:
    path = Path(__file__).parent / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""

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

CSV_FILE = "lab_attendance.csv"
STUDENT_REGISTRY_FILE = "student_registry.csv"

# Initialize CSV if it doesn't exist
if not os.path.exists(CSV_FILE):
    df_init = pd.DataFrame(columns=["Roll Number", "Date", "Time", "Lab"])
    df_init.to_csv(CSV_FILE, index=False)

# Initialize student registry if it doesn't exist
if not os.path.exists(STUDENT_REGISTRY_FILE):
    registry_init = pd.DataFrame(columns=["Roll Number", "Registration Date", "Face Encoding"])
    registry_init.to_csv(STUDENT_REGISTRY_FILE, index=False)

def load_data():
    return pd.read_csv(CSV_FILE)

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
    if day_of_week in [4, 5]:  # Friday or Saturday
        max_attendances = 2
    else:
        max_attendances = 1
    
    if existing_count >= max_attendances:
        existing_labs = existing_today["Lab"].unique()
        existing_labs_str = ", ".join(existing_labs)
        return False, f"You can't attend {lab} because you have already reached the maximum attendances ({max_attendances}) for today. Recorded for: {existing_labs_str}."
        
    new_record = pd.DataFrame([{
        "Roll Number": roll_number,
        "Date": current_date,
        "Time": current_time,
        "Lab": lab
    }])
    
    df = pd.concat([df, new_record], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    return True, f"Successfully marked attendance for {roll_number} in {lab} at {current_time}."

def load_student_registry():
    return pd.read_csv(STUDENT_REGISTRY_FILE)


def save_student_registry(df):
    df.to_csv(STUDENT_REGISTRY_FILE, index=False)


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


def enroll_student(roll_number):
    if not roll_number:
        return False, "Roll number cannot be empty."
    df = load_student_registry()
    if roll_number in df["Roll Number"].astype(str).tolist():
        return False, f"Roll number {roll_number} is already registered."
    
    new_record = pd.DataFrame([{
        "Roll Number": roll_number,
        "Registration Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Face Encoding": ""  # Kept empty for schema compatibility
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


def play_siri_voice(success, roll_number=""):
    """Play Siri voice note based on success/failure"""
    try:
        import platform
        import os
        system = platform.system()
        if system == 'Darwin':  # macOS
            roll_str = str(roll_number).replace('"', '').replace("'", "")
            prefix = f"{roll_str}, " if roll_str else ""
            if success:
                os.system(f'say -v alexa "{prefix}your attendance is successfully recorded." &')
            else:
                os.system(f'say -v alexa "{prefix}failed to record attendance." &')
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
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 80vh; text-align: center; font-family: 'Inter', sans-serif;">
            <div class="login-card" style="width: 450px; max-width: 90%; padding: 40px; border-radius: 20px; text-align: center; background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border: 1px solid rgba(255,255,255,0.04); box-shadow: 0 30px 80px rgba(2,6,23,0.7);">
                <div style="margin-bottom: 20px; font-size: 3.5rem;">🔑</div>
                <h2 style="color: #a5f3fc; text-shadow: 0 0 20px rgba(56, 189, 248, 0.3); font-weight: 800; font-size: 1.8rem; margin: 0 0 10px 0;">Signing in...</h2>
                <p style="color: rgba(230,238,248,0.8); margin-top: 10px; font-size: 1.1rem;">Welcome — redirecting {role_title} to attendance page...</p>
                <div style="margin-top: 30px; display: flex; justify-content: center;">
                    <div class="loader"></div>
                </div>
            </div>
        </div>
        <style>
            .loader {{
                border: 4px solid rgba(255, 255, 255, 0.05);
                width: 50px;
                height: 50px;
                border-radius: 50%;
                border-left-color: #38bdf8;
                border-top-color: #667eea;
                animation: spin 0.8s cubic-bezier(0.5, 0, 0.5, 1) infinite;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
        </style>
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
    # Render Application dashboard (student by default, faculty if logged in)
    col_logo, col_faculty_btn = st.columns([5, 1])
    with col_logo:
        st.markdown("<h1 style='margin:0; font-size: 2.2rem; display: flex; align-items: center;'>📝 Smart Lab Attendance</h1>", unsafe_allow_html=True)
    with col_faculty_btn:
        st.markdown("<div style='padding-top: 8px;'></div>", unsafe_allow_html=True)
        if st.session_state.user_role == "student":
            if st.button("🔐 Faculty Login", key="faculty_login_nav_btn", use_container_width=True):
                st.session_state.show_faculty_login = True
                st.rerun()
        else:
            if st.button("Logout", key="logout_btn", use_container_width=True):
                st.session_state.user_role = "student"
                st.session_state.username = "Student"
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
        
        st.markdown(
            """
            <div style="background: rgba(255,255,255,0.06); padding: 12px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.12); margin-top: 10px;">
                <span style="color: #38bdf8; font-weight: 700; font-size: 0.9rem;">💡 Testing Info:</span><br/>
                <span style="color: #dadde3; font-size: 0.85rem;">Email: <b style="color: #ffffff;">faculty@college.edu</b></span><br/>
                <span style="color: #dadde3; font-size: 0.85rem;">Password: <b style="color: #ffffff;">faculty123</b></span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("---")
            
    st.markdown(
        """
        <div class="hero" style="margin-top: 15px; padding: 1.5rem; border-radius: 20px; background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(148, 163, 184, 0.12);">
            <h2 style="color: #38bdf8 !important; text-shadow: none; font-size: 1.4rem; margin-top: 0; margin-bottom: 8px; font-weight: 700;">Secure Attendance Dashboard</h2>
            <p style="margin-bottom: 0; color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;">Scan QR codes using your camera feed, manage student registers, and export reports directly from your secure session.</p>
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
    if st.session_state.user_role == "student":
        tab_list = ["QR Code Scanner Attendance", "My QR Code"]
    else:
        tab_list = ["QR Code Scanner Attendance", "View Records", "Manage Students & QR Codes"]
        
    tabs = st.tabs(tab_list)
    
    with tabs[0]:
        st.header("📷 QR Code Scanner Attendance")
        lab_choice = st.radio("Select Subject:", ["Python", "Operating System", "Computer Graphics", "DataStructure", "Cpp", "DBMS", "DigitalLogics", "JAVA", "WebDesigen", "C Programing","R Programing"], horizontal=True)
        st.markdown("<div class='section-note'>Select the correct lab subject first, then present your QR Code to the camera or upload a QR image for attendance marking.</div>", unsafe_allow_html=True)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
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
            st.subheader("Manual Attendance")
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
            st.subheader("How to use QR Code Attendance")
            st.markdown("""
            - **Generate QR Code**: Enter your Roll Number and generate/download your personalized QR code.
            - **Scan or Upload**: Present your QR code to the webcam on the attendance page, or upload the saved image.
            - **Mark Attendance**: The system instantly reads the code, validates your roll number, and records your attendance.
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
            filter_lab = st.selectbox("Filter by Subject", ["All", "Python", "Operating System", "Computer Graphics", "DataStructure", "Cpp", "DBMS", "DigitalLogics", "JAVA", "WebDesigen", "C Programing"])
        
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
                            df = df.drop(idx_to_drop)
                            df.to_csv(CSV_FILE, index=False)
                            st.toast(f"✅ Deleted record for {record_to_delete.split(' - ')[0]}")
                            st.rerun()
            
            st.divider()
            st.subheader("Delete Records")
            st.write("Use these buttons to remove records for the selected date or the entire attendance history.")
            delete_date_button, delete_all_button = st.columns(2)
            with delete_date_button:
                if st.button(f"Delete all records for {selected_date.strftime('%b %d, %Y')}", key=f"delete_date_{selected_date_str}"):
                    df = df[df["Date"] != selected_date_str]
                    df.to_csv(CSV_FILE, index=False)
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
                            df = df.iloc[0:0]
                            df.to_csv(CSV_FILE, index=False)
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
            
            filter_lab_all = st.selectbox("Filter by Subject (All Records)", ["All", "Python", "Operating System", "Computer Graphics", "DataStructure", "Cpp", "DBMS", "DigitalLogics", "JAVA", "WebDesigen", "C Programing"], key="filter_all")
            
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
            st.header("🧑‍🎓 Student QR Code Registry")
            st.markdown("<div class='section-note'>Register new student Roll Numbers, list all registered students, and generate/download their personalized QR codes.</div>", unsafe_allow_html=True)
            
            st.subheader("Enroll New Student")
            enroll_roll = st.text_input("Enter Student Roll Number:", key="enroll_roll")
            if st.button("Register Student & Generate QR Code", type="primary", key="enroll_student_btn"):
                roll_clean = enroll_roll.strip()
                if roll_clean:
                    success, message = enroll_student(roll_clean)
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
                else:
                    st.warning("Please enter a valid Roll Number.")
            
            st.divider()
            st.subheader("Registered Students List")
            registry_df = load_student_registry()
            if not registry_df.empty:
                st.write(f"Total registered students: **{len(registry_df)}**")
                
                display_df = registry_df.copy()
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
