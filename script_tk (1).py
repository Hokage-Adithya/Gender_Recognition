import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import joblib
import wavio

# Constants
DURATION = 3  # seconds
SAMPLE_RATE = 22050
N_MFCC = 40

# Load model and label encoder
model = tf.keras.models.load_model("gender_model_limited.h5")
le = joblib.load("label_encoder.pkl")

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

def predict_gender(file_path):
    features = extract_features(file_path)
    if features is None:
        return "Error", 0.0
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features)
    label = le.inverse_transform([np.argmax(prediction)])[0]
    confidence = np.max(prediction)
    return label, confidence

def record_and_predict():
    filename = "realtime_input.wav"
    try:
        status_var.set("🎤 Recording audio...")
        root.update()
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()
        wavio.write(filename, audio, SAMPLE_RATE, sampwidth=2)
        label, confidence = predict_gender(filename)
        status_var.set(f"🎯 Predicted Gender: {label} ({confidence*100:.2f}%)")
    except Exception as e:
        messagebox.showerror("Recording Error", str(e))

def browse_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        label, confidence = predict_gender(file_path)
        status_var.set(f"🎯 Predicted Gender: {label} ({confidence*100:.2f}%)")

# GUI setup
root = tk.Tk()
root.title("Gender Detection Interface")
root.geometry("450x300")
root.configure(bg="#2C3E50")

style = ttk.Style()
style.theme_use("clam")
style.configure("TButton", font=("Helvetica", 12), padding=10, background="#3498DB", foreground="white")
style.configure("TLabel", font=("Helvetica", 14), background="#2C3E50", foreground="white")

main_frame = ttk.Frame(root, padding=20)
main_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

title = ttk.Label(main_frame, text="👤 Gender Prediction", font=("Helvetica", 18, "bold"))
title.grid(row=0, column=0, columnspan=2, pady=(0, 20))

btn_record = ttk.Button(main_frame, text="🎙 Real-time Recording", command=record_and_predict)
btn_record.grid(row=1, column=0, columnspan=2, pady=10)

btn_upload = ttk.Button(main_frame, text="📁 Upload .wav File", command=browse_and_predict)
btn_upload.grid(row=2, column=0, columnspan=2, pady=10)

status_var = tk.StringVar()
status_label = ttk.Label(main_frame, textvariable=status_var, wraplength=400, anchor="center", justify="center")
status_label.grid(row=3, column=0, columnspan=2, pady=20)

root.mainloop()
