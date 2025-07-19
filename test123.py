import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import queue
import os
import time
import random # For simulated prediction

# --- User's Provided Audio Processing Libraries ---
import pyaudio
import wave
import librosa
import numpy as np
from sys import byteorder
from array import array
from struct import pack

# --- Global Constants from User's Code ---
THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000 # Sample rate for recording and feature extraction

SILENCE = 30 # Number of silent chunks to detect before stopping recording

# --- Global variables for GUI state and audio file management ---
is_recording = False
current_audio_file_path = None # Path to the currently loaded or recorded audio file
recording_thread = None # To hold the recording thread reference
stop_recording_event = threading.Event() # Event to signal the recording thread to stop

# --- Dummy Model for Simulation ---
# In a real application, you would load your trained TensorFlow model here.
# For demonstration, we'll create a dummy class that simulates prediction.
class DummyModel:
    def predict(self, features):
        """Simulates model prediction."""
        # Simulate a probability for male (e.g., 0.0 to 1.0)
        male_prob = random.uniform(0.05, 0.95)
        # Return a structure similar to what a Keras model might return
        return np.array([[male_prob]])

# Initialize a dummy model
# In a real scenario, you'd do:
# from utils import create_model
# model = create_model()
# model.load_weights("results/model.h5")
model = DummyModel()

# --- User's Provided Audio Utility Functions ---
def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    # Avoid division by zero if snd_data is all zeros
    if not snd_data or max(abs(i) for i in snd_data) == 0:
        return array('h', [0 for _ in snd_data])

    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data_inner):
        snd_started = False
        r = array('h')

        for i in snd_data_inner:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)
            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for _ in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for _ in range(int(seconds*RATE))])
    return r

def record_audio_data(stop_event):
    """
    Record a word or words from the microphone and
    return the data as an array of signed shorts.
    This function is modified to be stoppable by an event.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
                    input=True, output=True,
                    frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False
    r = array('h')

    try:
        while not stop_event.is_set(): # Check stop_event to allow external stopping
            # little endian, signed short
            # Added exception_on_overflow=False to prevent blocking on buffer overflow
            snd_data = array('h', stream.read(CHUNK_SIZE, exception_on_overflow=False))
            if byteorder == 'big':
                snd_data.byteswap()
            r.extend(snd_data)

            silent = is_silent(snd_data)

            if silent and snd_started:
                num_silent += 1
            elif not silent and not snd_started:
                snd_started = True

            if snd_started and num_silent > SILENCE:
                break # Break if silence threshold met (voice activity detection)

    except IOError as e:
        # Handle input overflow or other PyAudio errors
        print(f"PyAudio error during recording: {e}")
        # Optionally, display an error message in the GUI
        root.after(0, lambda: status_label.config(text=f"Recording error: {e}", fg="red"))
    except Exception as e:
        print(f"An unexpected error occurred during recording: {e}")
        root.after(0, lambda: status_label.config(text=f"Unexpected recording error: {e}", fg="red"))
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    if r: # Only process if any data was recorded
        r = normalize(r)
        r = trim(r)
        r = add_silence(r, 0.5)
    return p.get_sample_size(FORMAT), r

def record_to_file(path, stop_event):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record_audio_data(stop_event)
    if not data: # If no data was recorded (e.g., stopped immediately or error)
        return False

    data_bytes = pack('<' + ('h'*len(data)), *data)

    try:
        wf = wave.open(path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(RATE)
        wf.writeframes(data_bytes)
        wf.close()
        return True
    except Exception as e:
        print(f"Error saving WAV file: {e}")
        root.after(0, lambda: messagebox.showerror("Save Error", f"Could not save recorded audio: {e}"))
        return False

def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")

    try:
        X, sample_rate = librosa.core.load(file_name, sr=RATE) # Ensure consistent sample rate
    except Exception as e:
        print(f"Error loading audio file for feature extraction: {e}")
        root.after(0, lambda: messagebox.showerror("Feature Extraction Error", f"Could not load audio file: {e}"))
        return None

    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    return result

# --- GUI Functions ---
def update_status(message, style_name="Status.Default.TLabel"):
    """Updates the status label in the GUI using a specific style."""
    status_label.config(text=message, style=style_name)

def set_gui_state(recording_active=False, audio_available=False, predicting=False):
    """Sets the state of GUI buttons based on current operation."""
    record_button.config(state=tk.DISABLED if recording_active else tk.NORMAL)
    stop_button.config(state=tk.NORMAL if recording_active else tk.DISABLED)
    upload_button.config(state=tk.DISABLED if recording_active or predicting else tk.NORMAL)
    play_button.config(state=tk.DISABLED if recording_active or predicting or not audio_available else tk.NORMAL)
    predict_button.config(state=tk.DISABLED if recording_active or predicting or not audio_available else tk.NORMAL)

def start_recording_gui():
    """Starts the audio recording process in a separate thread, updates GUI."""
    global is_recording, recording_thread, current_audio_file_path
    if is_recording:
        return

    is_recording = True
    stop_recording_event.clear() # Clear event for new recording
    current_audio_file_path = "test.wav" # Define temporary file path

    update_status("Recording... Speak now!", "Status.Info.TLabel")
    prediction_result_label.config(text="Prediction will appear here...")
    set_gui_state(recording_active=True)

    def record_and_save_thread():
        global is_recording, current_audio_file_path
        success = record_to_file(current_audio_file_path, stop_recording_event)
        is_recording = False # Recording thread finished

        root.after(0, lambda: handle_recording_finished(success))

    recording_thread = threading.Thread(target=record_and_save_thread, daemon=True)
    recording_thread.start()

def handle_recording_finished(success):
    """Callback after recording thread finishes."""
    global current_audio_file_path
    if success and os.path.exists(current_audio_file_path):
        update_status(f"Recording saved to {os.path.basename(current_audio_file_path)}", "Status.Success.TLabel")
        set_gui_state(audio_available=True)
    else:
        update_status("Recording stopped. No valid audio recorded or saved.", "Status.Warning.TLabel")
        current_audio_file_path = None
        set_gui_state(audio_available=False) # No audio available for play/predict

def stop_recording_gui():
    """Signals the recording thread to stop and updates GUI."""
    global is_recording
    if not is_recording:
        return

    stop_recording_event.set() # Signal the thread to stop
    update_status("Stopping recording...", "Status.Info.TLabel")
    # The `handle_recording_finished` will be called by the thread when it's done.

def play_audio_gui():
    """Plays the currently loaded or recorded audio file using PyAudio."""
    if not current_audio_file_path or not os.path.exists(current_audio_file_path):
        messagebox.showwarning("No Audio", "No audio file available to play. Record or upload first.")
        return

    update_status("Playing audio...", "Status.Info.TLabel")
    set_gui_state(predicting=True) # Disable buttons during playback

    def playback_thread():
        try:
            wf = wave.open(current_audio_file_path, 'rb')
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)

            data = wf.readframes(CHUNK_SIZE)
            while data:
                stream.write(data)
                data = wf.readframes(CHUNK_SIZE)

            stream.stop_stream()
            stream.close()
            p.terminate()
            wf.close()
            root.after(0, lambda: update_status("Audio playback finished.", "Status.Success.TLabel"))
        except Exception as e:
            root.after(0, lambda: messagebox.showerror("Playback Error", f"An error occurred during playback: {e}"))
            root.after(0, lambda: update_status("Playback failed.", "Status.Error.TLabel"))
        finally:
            root.after(0, lambda: set_gui_state(audio_available=True if current_audio_file_path else False))

    threading.Thread(target=playback_thread, daemon=True).start()

def upload_wav_file_gui():
    """Opens a file dialog to select a WAV file."""
    global current_audio_file_path
    file_path = filedialog.askopenfilename(
        title="Select .WAV File",
        filetypes=[("WAV files", "*.wav")]
    )
    if file_path:
        if os.path.exists(file_path):
            current_audio_file_path = file_path
            update_status(f"Loaded: {os.path.basename(file_path)}", "Status.Success.TLabel")
            prediction_result_label.config(text="Prediction will appear here...")
            set_gui_state(audio_available=True)
        else:
            messagebox.showerror("File Error", "Selected file does not exist.")
            update_status("File not found.", "Status.Error.TLabel")
            current_audio_file_path = None
            set_gui_state(audio_available=False)
    else:
        update_status("No file selected.", "Status.Warning.TLabel")
        current_audio_file_path = None
        set_gui_state(audio_available=False)

def make_prediction_gui():
    """Triggers feature extraction and simulated gender prediction."""
    if not current_audio_file_path:
        messagebox.showwarning("No Audio", "Please record audio or upload a .WAV file first.")
        return

    update_status("Extracting features and predicting...", "Status.Info.TLabel")
    prediction_result_label.config(text="Analyzing...")
    set_gui_state(predicting=True)

    def prediction_logic_thread():
        try:
            # Extract features using Mel Spectrogram as requested
            features = extract_feature(current_audio_file_path, mel=True)
            if features is None: # Handle error during feature extraction
                raise ValueError("Feature extraction failed.")

            # Reshape for model input (assuming model expects 2D array)
            features = features.reshape(1, -1)

            # Predict the gender using the dummy model
            male_prob = model.predict(features)[0][0]
            female_prob = 1 - male_prob
            gender = "Male" if male_prob > female_prob else "Female"

            # Update GUI with results
            root.after(0, lambda: [
                prediction_result_label.config(text=f"Predicted Gender: {gender}\n"
                                                    f"Male: {male_prob*100:.2f}% | Female: {female_prob*100:.2f}%", foreground="purple"), # Use foreground directly for this specific label
                update_status("Prediction complete.", "Status.Success.TLabel")
            ])
        except Exception as e:
            root.after(0, lambda: [
                messagebox.showerror("Prediction Error", f"An error occurred during prediction: {e}"),
                prediction_result_label.config(text="Prediction failed.", foreground="red"), # Use foreground directly for this specific label
                update_status("Prediction failed.", "Status.Error.TLabel")
            ])
        finally:
            root.after(0, lambda: set_gui_state(audio_available=True if current_audio_file_path else False))

    threading.Thread(target=prediction_logic_thread, daemon=True).start()

# --- GUI Setup ---
root = tk.Tk()
root.title("Gender Prediction")
root.geometry("500x600")
root.resizable(False, False)
root.configure(bg="#f0f4f8") # Light blue-gray background

# Styling
style = ttk.Style()
style.theme_use('clam') # 'clam' is a modern-looking theme
style.configure('TFrame', background='#f0f4f8')
style.configure('TLabel', background='#f0f4f8', font=('Inter', 12)) # Default TLabel style
style.configure('TButton', font=('Inter', 10, 'bold'), padding=10,
                background='#4f46e5', foreground='white', borderwidth=0, relief='flat')
style.map('TButton',
          background=[('active', '#4338ca'), ('disabled', '#cbd5e1')],
          foreground=[('disabled', '#64748b')])
style.configure('TLabelframe', background='#f0f4f8', foreground='#334155', font=('Inter', 14, 'bold'))
style.configure('TLabelframe.Label', background='#f0f4f8', foreground='#334155')

# Define specific styles for status messages
style.configure('Status.Default.TLabel', foreground="black")
style.configure('Status.Info.TLabel', foreground="blue")
style.configure('Status.Success.TLabel', foreground="green")
style.configure('Status.Warning.TLabel', foreground="orange")
style.configure('Status.Error.TLabel', foreground="red")


# Main Frame
main_frame = ttk.Frame(root, padding="20 20 20 20")
main_frame.pack(expand=True, fill='both')

# Title Label
title_label = ttk.Label(main_frame, text="Gender Prediction", font=('Inter', 24, 'bold'), foreground="#334155")
title_label.pack(pady=(0, 20))

# --- Record Audio Section ---
record_frame = ttk.LabelFrame(main_frame, text="Record Your Voice", padding="15 15 15 15")
record_frame.pack(pady=10, fill='x', padx=10)

record_button = ttk.Button(record_frame, text="Start Recording", command=start_recording_gui)
record_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

stop_button = ttk.Button(record_frame, text="Stop Recording", command=stop_recording_gui, state=tk.DISABLED)
stop_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

play_button = ttk.Button(record_frame, text="Play Audio", command=play_audio_gui, state=tk.DISABLED)
play_button.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

record_frame.grid_columnconfigure(0, weight=1)
record_frame.grid_columnconfigure(1, weight=1)


# --- Upload File Section ---
upload_frame = ttk.LabelFrame(main_frame, text="Upload .WAV File", padding="15 15 15 15")
upload_frame.pack(pady=10, fill='x', padx=10)

upload_button = ttk.Button(upload_frame, text="Choose .WAV File", command=upload_wav_file_gui)
upload_button.pack(pady=5, fill='x')


# --- Status and Prediction Section ---
# Initialize status_label with a default style
status_label = ttk.Label(main_frame, text="Ready", style="Status.Default.TLabel", font=('Inter', 10, 'italic'))
status_label.pack(pady=(10, 20))

predict_button = ttk.Button(main_frame, text="Predict Gender", command=make_prediction_gui, state=tk.DISABLED)
predict_button.pack(pady=10, fill='x', padx=10)

# Simplified prediction_result_label without custom style, using foreground directly for its color
prediction_result_label = ttk.Label(main_frame, text="Prediction will appear here...",
                                    font=('Inter', 16, 'bold'), background='#eef2ff', foreground='#4338ca',
                                    padding=15,
                                    anchor='center', wraplength=400)
prediction_result_label.pack(pady=(20, 10), fill='x', padx=10)


# --- Run the GUI ---
root.mainloop()

# --- Cleanup temporary file on exit (optional, but good practice) ---
if os.path.exists("test.wav"):
    try:
        os.remove("test.wav")
        print("Cleaned up test.wav")
    except Exception as e:
        print(f"Error cleaning up test.wav: {e}")