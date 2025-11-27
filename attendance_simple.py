import cv2
from datetime import datetime
import time
import sys

# Haarcascade file for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

marked = False
start_time = time.time()   # for safety auto-exit

def mark_attendance(name="User"):
    global marked
    if marked:
        return

    marked = True
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    with open("attendance.csv", "a") as f:
        f.write(f"{name},{date},{time_str}\n")

    print(f"Attendance marked for {name} at {time_str}")

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Face Detected", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        mark_attendance("User")

    cv2.imshow("Face Attendance System", frame)

    # 1) Press q to exit
    # 2) OR auto-exit after 15 seconds (safety)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("q pressed, exiting loop")
        break

    if time.time() - start_time > 15:
        print("Auto exit after 15 seconds")
        break

cap.release()
cv2.destroyAllWindows()
print("Program finished. Close VS Code terminal if you want.")
sys.exit()
