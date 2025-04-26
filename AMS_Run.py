import tkinter as tk
from tkinter import messagebox
import cv2
import os
import numpy as np
import pandas as pd
import datetime
from PIL import Image

# Create necessary directories
for folder in ["TrainingImage", "TrainingImageLabel", "Attendance", "StudentDetails"]:
    if not os.path.exists(folder):
        os.makedirs(folder)

window = tk.Tk()
window.title("Face Recognition Attendance System")
window.geometry('900x700')
window.configure(background='light green')

# Live Clock
def update_clock():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clock_label.config(text=now)
    window.after(1000, update_clock)

clock_label = tk.Label(window, font=('Helvetica', 12), bg='light green')
clock_label.pack()
update_clock()

# Take Images
def take_images():
    enrollment = enrollment_entry.get()
    name = name_entry.get()

    if enrollment == "" or name == "":
        messagebox.showerror("Error", "Please enter Enrollment and Name")
        return

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if detector.empty():
        messagebox.showerror("Error", "Haarcascade file not found.")
        return

    sample_num = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sample_num += 1
            cv2.imwrite(f"TrainingImage/{name}.{enrollment}.{sample_num}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, f"{name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        cv2.imshow('Capturing Images', img)
        if cv2.waitKey(1) & 0xFF == ord('q') or sample_num >= 70:
            break

    cam.release()
    cv2.destroyAllWindows()

    # Save to CSV
    df = pd.DataFrame([[enrollment, name]], columns=['Enrollment', 'Name'])
    path = "StudentDetails/StudentDetails.csv"
    if os.path.exists(path):
        existing = pd.read_csv(path)
        if enrollment not in existing['Enrollment'].astype(str).values:
            df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(path, index=False)
    messagebox.showinfo("Success", f"Images saved for {name} ({enrollment})")

# Train Model
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if detector.empty():
        messagebox.showerror("Error", "Haarcascade file not found.")
        return

    faces, ids = [], []
    path = "TrainingImage"
    if not os.listdir(path):
        messagebox.showwarning("Warning", "No images to train.")
        return

    for image_path in os.listdir(path):
        img = Image.open(os.path.join(path, image_path)).convert('L')
        img_np = np.array(img, 'uint8')
        id = int(image_path.split(".")[1])
        faces_detected = detector.detectMultiScale(img_np)
        for (x, y, w, h) in faces_detected:
            faces.append(img_np[y:y+h, x:x+w])
            ids.append(id)

    recognizer.train(faces, np.array(ids))
    recognizer.save("TrainingImageLabel/Trainner.yml")
    messagebox.showinfo("Success", "Model trained successfully.")

# Recognize Faces and Mark Attendance
def recognize_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if detector.empty():
        messagebox.showerror("Error", "Haarcascade file not found.")
        return

    marked_ids = set()
    cam = cv2.VideoCapture(0)

    runanalysis = True
    while runanalysis:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            if confidence > 90:
                if id not in marked_ids:
                    try:
                        df = pd.read_csv("StudentDetails/StudentDetails.csv")
                        name = df.loc[df['Enrollment'] == id, 'Name'].values[0]
                        attendance_record = f"{datetime.datetime.now().strftime('%Y-%m-%d')},{id},{name}\n"
                        with open(f"Attendance/{datetime.datetime.now().strftime('%Y-%m-%d')}.csv", 'a') as f:
                            f.write(attendance_record)
                        marked_ids.add(id)
                        runanalysis = False
                    except IndexError:
                        name = "Unknown"
                        print(f"Unknown ID {id} not found in the records.")
                confidence_text = f"  {round(100 - confidence)}%"
            else:
                name = "Unknown"
                confidence_text = f"  {round(100 - confidence)}%"

            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, f"{name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            cv2.putText(img, confidence_text, (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        cv2.imshow('Recognizing Faces', img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == 27:  # Esc key
            break
        elif key == 13:  # Enter key
            break

    cam.release()
    cv2.destroyAllWindows()

# View Attendance Records
def view_attendance_records():
    folder = "Attendance"
    files = os.listdir(folder)
    if not files:
        messagebox.showinfo("Info", "No attendance records found.")
        return

    top = tk.Toplevel(window)
    top.title("Attendance Records")
    top.geometry("500x400")

    listbox = tk.Listbox(top, font=('Helvetica', 12))
    for file in files:
        listbox.insert('end', file)
    listbox.pack(fill='both', expand=True)

    def open_file():
        selected = listbox.get(listbox.curselection())
        if not selected.endswith('.csv'):
            messagebox.showerror("Error", "Selected file is not a CSV file.")
            return
        try:
            df = pd.read_csv(f"{folder}/{selected}")
            view = tk.Toplevel(top)
            view.title(selected)
            text = tk.Text(view)
            text.pack(fill='both', expand=True)
            text.insert('end', df.to_string(index=False))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    tk.Button(top, text="Open Selected", command=open_file).pack(pady=10)

# Delete Images
def delete_images():
    """Delete all images in the TrainingImage directory."""
    if messagebox.askyesno("Confirm Deletion", "Are you sure you want to delete all images?"):
        folder = "TrainingImage"
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete {filename}: {str(e)}")
        messagebox.showinfo("Success", "All images deleted successfully.")

# UI Elements
enrollment_label = tk.Label(window, text="Enrollment No:", bg='light green')
enrollment_label.pack(pady=10)
enrollment_entry = tk.Entry(window)
enrollment_entry.pack(pady=10)


def takelastAttendance():
    date = datetime.datetime.now().strftime('%Y-%m-%d')
    file_path = "Attendance/" + date + ".csv"
    if not os.path.exists(file_path):
        return ""
    attendance = pd.read_csv(file_path)
    if attendance.empty:
        return ""
    last = attendance.iloc[-1]
    return "Last Attendance:\n" + last.to_string(index=False)

enrollment_label1 = tk.Label(window, text= takelastAttendance(), bg='light green')
enrollment_label1.pack(pady=10)

name_label = tk.Label(window, text="Name:", bg='light green')
name_label.pack(pady=10)
name_entry = tk.Entry(window)
name_entry.pack(pady=10)

take_images_button = tk.Button(window, text="Take Images", command=take_images)
take_images_button.pack(pady=10)

train_model_button = tk.Button(window, text="Train Model", command=train_model)
train_model_button.pack(pady=10)

recognize_faces_button = tk.Button(window, text="Recognize Faces", command=recognize_faces)
recognize_faces_button.pack(pady=10)

view_attendance_button = tk.Button(window, text="View Attendance Records", command=view_attendance_records)
view_attendance_button.pack(pady=10)

delete_images_button = tk.Button(window, text="Delete Images", command=delete_images)
delete_images_button.pack(pady=10)

# Start the GUI window
window.mainloop()