import os
import sqlite3
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

PDF_FILENAME = "Smart_Lab_Attendance_Project_Documentation.pdf"

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#71717a"))
        
        # Suppress headers/footers on page 1 (cover header)
        if self._pageNumber > 1:
            # Header
            self.drawString(54, 11 * 72 - 36, "Smart Lab Attendance System — Complete Project Documentation")
            self.setStrokeColor(colors.HexColor("#e4e4e7"))
            self.setLineWidth(0.5)
            self.line(54, 11 * 72 - 42, 8.5 * 72 - 54, 11 * 72 - 42)
            
            # Footer
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(8.5 * 72 - 54, 36, page_text)
            self.drawString(54, 36, "Confidential & Proprietary — Department of Computer Applications")
            self.line(54, 46, 8.5 * 72 - 54, 46)
            
        self.restoreState()

def build_pdf():
    doc = SimpleDocTemplate(
        PDF_FILENAME,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#09090b"),
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#52525b"),
        spaceAfter=20
    )

    meta_style = ParagraphStyle(
        'CoverMeta',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2563eb")
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#09090b"),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#18181b"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#27272a"),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#27272a"),
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=4
    )

    code_style = ParagraphStyle(
        'Code_Custom',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#09090b"),
        backColor=colors.HexColor("#f4f4f5"),
        borderColor=colors.HexColor("#e4e4e7"),
        borderWidth=0.5,
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=6
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#18181b")
    )

    story = []

    # ==================== COVER HEADER ====================
    story.append(Paragraph("Smart Lab Attendance System", title_style))
    story.append(Paragraph("Comprehensive Project Documentation, System Architecture, Database Schema, and Working Process", subtitle_style))
    story.append(Paragraph("PROJECT REPORT & TECHNICAL SPECIFICATION &bull; JULY 2026", meta_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#09090b"), spaceBefore=8, spaceAfter=14))

    # ==================== SECTION 1: EXECUTIVE SUMMARY ====================
    story.append(Paragraph("1. Executive Summary & Overview", h1_style))
    story.append(Paragraph(
        "The <b>Smart Lab Attendance System</b> is an autonomous, multi-modal attendance verification and analytics platform designed specifically for academic computer laboratories and university departments. Built using Python, Streamlit, OpenCV, and SQLite, the system replaces manual roll-call registers with AI-driven face recognition, anti-spoofing liveness detection, and GPS geofencing.",
        body_style
    ))
    story.append(Paragraph(
        "The system enforces strict security protocols: students must be physically present within the college perimeter (verified via HTML5 Geolocation API and Geopy distance calculation) and pass a real-time eye-blink liveness challenge before attendance is marked. Additionally, administrators and faculty members can export structured attendance reports grouped by faculty member, subject, year, and semester as individual CSV files or bulk ZIP archives.",
        body_style
    ))

    # ==================== SECTION 2: SYSTEM ARCHITECTURE ====================
    story.append(Paragraph("2. System Architecture & Tech Stack", h1_style))
    story.append(Paragraph("The system is structured into five core technological layers:", body_style))

    tech_data = [
        [Paragraph("Layer", table_header_style), Paragraph("Technology / Library", table_header_style), Paragraph("Function & Purpose", table_header_style)],
        [Paragraph("Frontend UI", table_cell_style), Paragraph("Streamlit + Custom CSS (Inter)", table_cell_style), Paragraph("Minimalist responsive web dashboard with role-based routing.", table_cell_style)],
        [Paragraph("Computer Vision & AI", table_cell_style), Paragraph("OpenCV + Face Recognition (Dlib)", table_cell_style), Paragraph("HOG face detection, 68-point facial landmarks, and 128D feature encodings.", table_cell_style)],
        [Paragraph("Geofencing & Security", table_cell_style), Paragraph("Geopy + Streamlit JS Eval", table_cell_style), Paragraph("HTML5 browser GPS extraction & geodesic distance calculation (40m radius lock).", table_cell_style)],
        [Paragraph("Database Storage", table_cell_style), Paragraph("SQLite3 (Embedded Relational DB)", table_cell_style), Paragraph("Zero-config transactional storage for students, faculty, and attendance logs.", table_cell_style)],
        [Paragraph("Audio & Voice", table_cell_style), Paragraph("macOS Siri Speech Engine (say -v siri)", table_cell_style), Paragraph("Real-time spoken audio feedback confirming attendance status.", table_cell_style)]
    ]

    t_tech = Table(tech_data, colWidths=[110, 160, 234])
    t_tech.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#18181b")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e4e4e7")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_tech)
    story.append(Spacer(1, 10))

    # ==================== SECTION 3: DATABASE ARCHITECTURE ====================
    story.append(Paragraph("3. Database Architecture & Schema (SQLite)", h1_style))
    story.append(Paragraph(
        "All application state and historical records are stored in a local SQLite database named <b><code>attendance.db</code></b>. On startup, the system automatically initializes three relational tables and migrates legacy CSV files if present.",
        body_style
    ))

    story.append(Paragraph("A. Table Schemas", h2_style))

    db_schema_data = [
        [Paragraph("Table Name", table_header_style), Paragraph("Columns & Types", table_header_style), Paragraph("Description & Constraints", table_header_style)],
        [
            Paragraph("<b>attendance</b>", table_cell_style),
            Paragraph("id INTEGER PRIMARY KEY AUTOINCREMENT<br/>roll_number TEXT NOT NULL<br/>date TEXT NOT NULL<br/>time TEXT NOT NULL<br/>lab TEXT NOT NULL", table_cell_style),
            Paragraph("Logs every valid attendance entry marked by students via Face Recognition, QR Code, or Manual Bypass.", table_cell_style)
        ],
        [
            Paragraph("<b>students</b>", table_cell_style),
            Paragraph("roll_number TEXT PRIMARY KEY<br/>registration_date TEXT<br/>face_encoding TEXT (JSON string)<br/>face_path TEXT", table_cell_style),
            Paragraph("Stores registered student Roll Numbers, 128-dimensional face encoding vectors (serialized as JSON), and face image filepaths.", table_cell_style)
        ],
        [
            Paragraph("<b>faculties</b>", table_cell_style),
            Paragraph("id INTEGER PRIMARY KEY AUTOINCREMENT<br/>name TEXT NOT NULL<br/>email TEXT NOT NULL<br/>department TEXT NOT NULL<br/>year TEXT<br/>semester TEXT", table_cell_style),
            Paragraph("Stores faculty registrations mapped to specific subjects, academic years (1st, 2nd, 3rd Year), and semesters (1st to 6th Sem).", table_cell_style)
        ]
    ]

    t_db = Table(db_schema_data, colWidths=[90, 210, 204])
    t_db.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#18181b")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e4e4e7")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_db)
    story.append(Spacer(1, 10))

    # ==================== SECTION 4: WORKING PROCESS & WORKFLOW ====================
    story.append(Paragraph("4. Working Process & End-to-End Workflow", h1_style))

    workflow_steps = [
        "<b>Step 1: Geofencing Check</b> — When the user opens the application, JavaScript extracts browser latitude/longitude coordinates via HTML5 Geolocation API. Geopy computes distance to college location <i>(15.2738&deg; N, 76.3774&deg; E)</i>. Access is blocked if the distance exceeds 40 meters.",
        "<b>Step 2: Role-Based Dashboard Routing</b> — Users interact with the app based on active session state:<br/>&bull; <i>Student Mode</i>: Direct access to Face Scanner, QR Scanner, Face Registration, and Personal QR Code Generator.<br/>&bull; <i>Faculty Mode</i>: View, search, filter daily lab attendance records, and delete records.<br/>&bull; <i>Admin Mode</i>: Manage Faculty registry, Bulk/Per-Faculty CSV and ZIP downloads filtered by Subject and Semester, Attendance Analytics, and Manage Records.",
        "<b>Step 3: Student Enrollment</b> — When a student registers, their webcam/uploaded photo is analyzed by Dlib's facial recognition algorithm. A 128-dimensional floating-point encoding array is serialized into a JSON string and saved into the <code>students</code> SQLite table along with a JPEG snapshot saved under <code>registered_faces/<roll_number>.jpg</code>.",
        "<b>Step 4: Liveness Detection & Face Verification</b> — The camera feed captures frames continuously:<br/>1. Facial landmarks are extracted for left eye (landmarks 36-41) and right eye (landmarks 42-47).<br/>2. Eye Aspect Ratio (EAR) is calculated: <code>EAR = (|p1 - p5| + |p2 - p4|) / (2 * |p0 - p3|)</code>.<br/>3. The student is challenged to <b>blink their eyes</b>. A valid blink transition (open &rarr; closed &rarr; reopened) proves live presence, preventing spoofing via photos or smartphone screens.",
        "<b>Step 5: Attendance Rules & Recording</b> — Once liveness and face match are confirmed:<br/>&bull; Attendance is inserted into the <code>attendance</code> SQLite table.<br/>&bull; Daily attendance cap is enforced (max 1 session on Mon-Thu, max 2 sessions on Fri-Sat).<br/>&bull; Spoken audio feedback is triggered using macOS Siri speech synthesis.",
        "<b>Step 6: Admin Attendance Export</b> — Admin selects date ranges, year, semester, or specific faculty. The system generates customized CSV reports per faculty member or a structured ZIP archive grouped by <code>Year/Semester/Faculty_Subject_attendance.csv</code>."
    ]

    for step in workflow_steps:
        story.append(Paragraph(f"&bull; {step}", bullet_style))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 8))

    # ==================== SECTION 5: FILE-BY-FILE EXPLANATION ====================
    story.append(Paragraph("5. Detailed File-by-File Breakdown", h1_style))

    file_breakdown = [
        [Paragraph("File Name", table_header_style), Paragraph("Role & Responsibilities in Project", table_header_style)],
        [
            Paragraph("<b>attend.py</b>", table_cell_style),
            Paragraph("Main Streamlit application (1,900+ lines of Python). Contains all business logic, SQLite helper functions (<code>init_db</code>, <code>load_data</code>, <code>mark_attendance</code>, <code>enroll_student</code>), OpenCV webcam loop, EAR blink liveness detection engine, and multi-tab UI rendering.", table_cell_style)
        ],
        [
            Paragraph("<b>style.css</b>", table_cell_style),
            Paragraph("Custom CSS stylesheet delivering a sleek Minimalist dark theme (`#09090b` obsidian background), Inter font typography, custom Streamlit button styling, metric card containers, and modal popups.", table_cell_style)
        ],
        [
            Paragraph("<b>attendance.db</b>", table_cell_style),
            Paragraph("SQLite relational database file storing relational tables for <code>attendance</code>, <code>students</code>, and <code>faculties</code>.", table_cell_style)
        ],
        [
            Paragraph("<b>registered_faces/</b>", table_cell_style),
            Paragraph("Directory storing JPEG image snapshots of registered student faces, named as <code><roll_number>.jpg</code>.", table_cell_style)
        ],
        [
            Paragraph("<b>qrcodes/</b> & <b>qrcode_saver.py</b>", table_cell_style),
            Paragraph("Directory and utility script for generating and storing downloadable PNG QR code images for student roll numbers.", table_cell_style)
        ],
        [
            Paragraph("<b>requirements.txt</b>", table_cell_style),
            Paragraph("Dependency manifest file specifying required packages: <code>streamlit</code>, <code>face_recognition</code>, <code>opencv-python</code>, <code>pandas</code>, <code>geopy</code>, <code>streamlit_js_eval</code>, <code>qrcode</code>, <code>Pillow</code>.", table_cell_style)
        ]
    ]

    t_files = Table(file_breakdown, colWidths=[130, 374])
    t_files.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#18181b")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e4e4e7")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_files)
    story.append(Spacer(1, 10))

    # ==================== SECTION 6: SERVERS & DATABASE ====================
    story.append(Paragraph("6. Servers & Database Architecture", h1_style))

    story.append(Paragraph("A. Database Engine (SQLite)", h2_style))
    story.append(Paragraph(
        "The project utilizes <b>SQLite3</b>, an embedded SQL database engine. SQLite requires zero server configuration, operates directly on local file storage (<code>attendance.db</code>), and provides full ACID compliance with high read/write performance suitable for local academic lab deployments.",
        body_style
    ))

    story.append(Paragraph("B. Server & Deployment Options", h2_style))
    story.append(Paragraph("The application can be hosted across multiple server environments:", body_style))
    
    server_points = [
        "<b>Local Server (Current)</b>: Launched via <code>streamlit run attend.py</code> serving on <code>http://localhost:8501</code> (or 8503) accessible within campus LAN.",
        "<b>Cloud VPS / EC2 Deployment</b>: Can be hosted on Ubuntu Linux (AWS EC2 / DigitalOcean) behind an Nginx Reverse Proxy with SSL (HTTPS) to allow remote access.",
        "<b>Enterprise Scaling (PostgreSQL / MySQL Migration)</b>: For multi-building or multi-campus deployments, SQLite database calls in <code>attend.py</code> can be swapped to PostgreSQL / MySQL with central server connectivity."
    ]
    for sp in server_points:
        story.append(Paragraph(f"&bull; {sp}", bullet_style))

    story.append(Spacer(1, 10))

    # ==================== SECTION 7: ADVANTAGES & DISADVANTAGES ====================
    story.append(Paragraph("7. Advantages & Disadvantages", h1_style))

    adv_dis_data = [
        [Paragraph("Advantages (&check;)", table_header_style), Paragraph("Disadvantages & Limitations (&cross;)", table_header_style)],
        [
            Paragraph(
                "&bull; <b>Anti-Spoofing Security</b>: Real-time Eye Aspect Ratio (EAR) blink detection prevents photo/screen proxy attacks.<br/>"
                "&bull; <b>Geofenced Attendance</b>: GPS radius lock ensures students must be inside college premise.<br/>"
                "&bull; <b>Granular Reports</b>: Download attendance filtered per faculty, per subject, and per semester as CSV or ZIP.<br/>"
                "&bull; <b>Zero DB Server Setup</b>: SQLite embedded database operates seamlessly without external database servers.<br/>"
                "&bull; <b>Multi-Modal Flexibility</b>: Supports Face Recognition, QR Code Scanning, and Manual Faculty Bypass.",
                table_cell_style
            ),
            Paragraph(
                "&bull; <b>Webcam Hardware Requirement</b>: Requires a functional camera on user devices.<br/>"
                "&bull; <b>Browser Geolocation Permission</b>: Users must grant location permissions in browser settings.<br/>"
                "&bull; <b>Indoor GPS Signal Limits</b>: Deep basement computer labs may experience weak GPS signals.<br/>"
                "&bull; <b>Single Server File Locking</b>: SQLite is optimal for single-server setups; multi-server clusters require PostgreSQL.",
                table_cell_style
            )
        ]
    ]

    t_adv = Table(adv_dis_data, colWidths=[252, 252])
    t_adv.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#18181b")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e4e4e7")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_adv)

    # Build PDF with page numbers
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✅ PDF documentation generated successfully: {PDF_FILENAME}")

if __name__ == "__main__":
    build_pdf()
