from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def build_pdf():
    pdf_filename = "Docker_DB_and_Operations_Guide.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=30,
        rightMargin=30,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontSize=15,
        leading=18,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=10
    )

    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=10,
        spaceAfter=6
    )

    cell_style = ParagraphStyle(
        'Cell',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#334155')
    )

    elements = []

    # Title
    elements.append(Paragraph("AI Face Attendance: Daily Operations & Verification Guide", title_style))
    elements.append(Spacer(1, 4))

    # Table 1: Daily Lifecycle
    elements.append(Paragraph("1. Daily Operations Lifecycle (Start se Lekar Stop Tak)", h2_style))
    ops_rows = [
        ["Phase", "Command", "Purpose & Explanation"],
        ["1. Folder", "cd ~/Music/AI_Face_recoginition/backend", "Project directory me enter karne ke liye."],
        ["2. Venv", "source venv/bin/activate", "Python virtual environment activate karta hai."],
        ["3. Containers", "docker compose up -d", "MySQL (3306) aur Typesense (8109) background me run karta hai."],
        ["4. Run App", "uvicorn main:app --host 0.0.0.0 --port 8000 --reload", "AI Face detector, webcam loop aur FastAPI server chalu karta hai."],
        ["5. Stop App", "Ctrl + C (Terminal) / 'q' (Camera)", "Camera hardware release karke Python server stop karta hai."],
        ["6. Stop Docker", "docker compose down", "Containers safely stop karke RAM aur ports free karta hai."],
        ["7. Exit Venv", "deactivate", "Virtual environment close karne ke liye."]
    ]

    t1_data = []
    for row in ops_rows:
        t1_data.append([
            Paragraph(row[0], cell_style),
            Paragraph(f"<b>{row[1]}</b>", cell_style),
            Paragraph(row[2], cell_style)
        ])

    t1 = Table(t1_data, colWidths=[65, 230, 255])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(t1)
    elements.append(Spacer(1, 10))

    # Table 2: Database Verification
    elements.append(Paragraph("2. Docker Database & Tables Check Commands", h2_style))
    db_rows = [
        ["Target", "Command", "Explanation"],
        ["Direct MySQL", "docker exec -it attendance_mysql mysql -u ai_attendance -p2003 -e 'SHOW DATABASES; USE Ai_attendance; SHOW TABLES;'", "Confirm karta hai ki Ai_attendance DB aur attendance_logs table bani hai."],
        ["Interactive Shell", "docker exec -it attendance_mysql mysql -u ai_attendance -p2003", "Container ke andar MySQL console (mysql>) open karta hai manual queries ke liye."],
        ["Table Schema", "docker exec -it attendance_mysql mysql -u ai_attendance -p2003 -e 'DESCRIBE Ai_attendance.attendance_logs;'", "Columns: id, user_id, name, login_time, logout_time, duration verify karta hai."],
        ["Live Attendance", "docker exec -it attendance_mysql mysql -u ai_attendance -p2003 -e 'SELECT * FROM Ai_attendance.attendance_logs;'", "Camera dwara dynamically mark kiye gaye live login/logout records dikhata hai."],
        ["Typesense Vector", "curl -H 'X-TYPESENSE-API-KEY: xyz123' http://localhost:8109/collections", "face_embeddings collection aur 512-dim vectors verify karta hai."]
    ]

    t2_data = []
    for row in db_rows:
        t2_data.append([
            Paragraph(row[0], cell_style),
            Paragraph(f"<b>{row[1]}</b>", cell_style),
            Paragraph(row[2], cell_style)
        ])

    t2 = Table(t2_data, colWidths=[90, 235, 225])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(t2)
    elements.append(Spacer(1, 10))

    # Summary
    elements.append(Paragraph("3. Daily One-Liner Shortcuts", h2_style))
    summary_text = (
        "<b>Fast Start:</b> <font color='#0F766E'>cd ~/Music/AI_Face_recoginition/backend && source venv/bin/activate && docker compose up -d && uvicorn main:app --host 0.0.0.0 --port 8000 --reload</font><br/>"
        "<b>Safe Stop:</b> Terminal me Ctrl+C dabayein, fir: <font color='#0F766E'>docker compose down && deactivate</font><br/>"
        "<b>Browser Check:</b> <font color='#0F766E'>http://localhost:8000/attendance/logs</font>"
    )
    elements.append(Paragraph(summary_text, cell_style))

    doc.build(elements)
    print("\n>>> SUCCESS: Docker_DB_and_Operations_Guide.pdf is READY! <<<\n")

if __name__ == "__main__":
    build_pdf()