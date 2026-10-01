pdf_data = """%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj
4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
5 0 obj << /Length 6 0 R >>
stream
BT
/F1 14 Tf
50 740 Td
(AI Face Recognition - Front Cam Login/Logout System Log) Tj
/F1 9.5 Tf
0 -28 Td
(1. Architecture: Single Front Camera (Index 0) automated authentication pipeline) Tj
0 -20 Td
(2. Core Stack: PyTorch, facenet-pytorch (MTCNN + ResNet), OpenCV, NumPy, Pillow) Tj
0 -20 Td
(3. First Arrival: User detected for the first time triggers automated LOGIN) Tj
0 -20 Td
(4. Cooldown Mechanism: 10-second safety buffer prevents false immediate triggers) Tj
0 -20 Td
(5. Re-entry Action: After cooldown, returning user automatically triggers LOGOUT) Tj
0 -20 Td
(6. Dynamic Purging: Previous image cropped file auto-purged, updated with latest frame) Tj
0 -20 Td
(7. Activity Tracking: Every Login and Logout is appended to attendance_log.csv) Tj
0 -20 Td
(8. Visual Feedback: Green box on screen for LOGIN, Red box overlay for LOGOUT) Tj
ET
endstream
endobj
6 0 obj
710
endobj
xref
0 7
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000222 00000 n 
0000000293 00000 n 
0000001075 00000 n 
trailer << /Size 7 /Root 1 0 R >>
startxref
1096
%%EOF"""

with open("AI_Face_Recognition_Full_Notes.pdf", "wb") as f:
    f.write(pdf_data.encode("latin1"))

print("[+] PDF successfully updated with Front Camera Login/Logout details!")