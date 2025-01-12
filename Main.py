import tkinter as tk
from PIL import Image, ImageTk
from threading import Thread, Event
import time
from Yolo import FaceRecognition
from Voice import (
    load_speaker_model,
    predict_speaker,
    record_audio,
)
import cv2

class Application:
    def __init__(self, root):
        self.root = root
        self.root.title("Face and Voice Recognition")
        self.face_recognition = None
        self.face_recognition_thread = None
        self.detected_face_label = None
        self.voice_model = None
        self.voice_recognition_result = None
        self.label_to_speaker = {0: "Alfred", 1: "Gilbert", 2: "Janssen", 3: "Noise"}

        # Initialize stop_event
        self.stop_event = Event()  # Add this line to initialize stop_event

        # GUI Elements
        self.info_label = tk.Label(root, text="Press 'Start' to begin face recognition.", font=("Arial", 14))
        self.info_label.pack(pady=20)

        self.start_btn = tk.Button(root, text="Start Face Recognition", command=self.start_face_recognition)
        self.start_btn.pack(pady=10)

        self.done_btn = tk.Button(root, text="Done", command=self.stop_face_recognition, state=tk.DISABLED)
        self.done_btn.pack(pady=10)

        self.countdown_label = tk.Label(root, text="", font=("Arial", 20))
        self.countdown_label.pack(pady=20)

        self.result_label = tk.Label(root, text="", font=("Arial", 16), fg="green")
        self.result_label.pack(pady=20)

        # Load the speaker identification model
        model_path = r"C:\Users\User\OneDrive\skull\S7\Citra_dan_suara\YOLO\APP\speaker_identification_model.h5"
        self.voice_model = load_speaker_model(model_path)

    def start_face_recognition(self):
        """Start face recognition and video stream in a separate thread."""
        self.info_label.config(text="Recognizing face... Please wait.")
        model_path = r"C:\Users\User\OneDrive\skull\S7\Citra_dan_suara\YOLO\APP\last.pt"
        self.face_recognition = FaceRecognition(model_path)

        # Create a new thread for running face recognition
        self.face_recognition_thread = Thread(target=self.run_face_recognition)
        self.face_recognition_thread.start()

        self.start_btn.config(state=tk.DISABLED)
        self.done_btn.config(state=tk.NORMAL)

    def run_face_recognition(self):
        """Run face recognition in a thread and show video in a new window."""
        cap = cv2.VideoCapture(0)  # Open the webcam

        if not cap.isOpened():
            print("Error: Could not open video stream.")
            return

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame and detect faces
            frame = self.face_recognition.process_frame(frame)

            # Capture the first detected label (if any)
            if self.face_recognition.detected_labels:
                self.detected_face_label = list(self.face_recognition.detected_labels)[0]  # Update label to the first detected face
            else:
                self.detected_face_label = "Unknown"  # If no face detected

            # Display the frame in the OpenCV window
            cv2.imshow("Face Recognition Video", frame)

            # Store the last frame to display it on the Tkinter window later
            self.last_frame = frame
            cv2.waitKey(1)

        # After the loop ends, release the video capture and close the window
        cap.release()
        cv2.destroyAllWindows()  # Close the OpenCV window



    def stop_face_recognition(self):
        """Stop the face recognition video stream and show last frame on main window."""
        if self.face_recognition:
            # Set the stop_event to stop the video capture thread
            self.stop_event.set()

            # Wait for the face recognition thread to finish
            self.face_recognition_thread.join()

            # Show the last frame captured from the video stream
            self.show_last_frame_on_main_window()

            # Update the GUI text to reflect face recognition result
            self.info_label.config(text=f"Face Recognized: {self.detected_face_label}")
            self.done_btn.config(state=tk.DISABLED)
            
            # Start countdown for voice recognition
            self.start_countdown()

    def show_last_frame_on_main_window(self):
        """Convert the last frame to a PhotoImage and display it on the main window."""
        # Convert the frame from BGR (OpenCV format) to RGB
        last_frame_rgb = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
        
        # Convert the NumPy array to a PIL Image
        pil_image = Image.fromarray(last_frame_rgb)
        
        # Convert the PIL Image to a Tkinter PhotoImage
        last_frame_image = ImageTk.PhotoImage(pil_image)

        # Display the image on the main window
        self.last_frame_label = tk.Label(self.root, image=last_frame_image)
        self.last_frame_label.image = last_frame_image  # Keep a reference to avoid garbage collection
        self.last_frame_label.pack(pady=20)


    def start_countdown(self):
        """Start a countdown for voice recognition."""
        def countdown():
            for i in range(3, 0, -1):
                self.countdown_label.config(text=f"Voice Recognition in {i}...")
                time.sleep(1)
            self.countdown_label.config(text="Recording...")
            self.start_voice_recognition()

        Thread(target=countdown).start()

    def start_voice_recognition(self):
        """Perform voice recognition."""
        duration = 3
        sample_rate = 16000
        num_mfcc = 13

        # Record audio
        audio = record_audio(duration, sample_rate)
        self.countdown_label.config(text="Recording Complete")

        # Predict speaker
        speaker, confidence = predict_speaker(audio, self.voice_model, self.label_to_speaker, sample_rate, num_mfcc)
        self.voice_recognition_result = speaker
        self.check_match(speaker)

    def check_match(self, speaker):
        """Check if face recognition matches voice recognition and update the GUI."""
        self.countdown_label.config(text="")

        # Update the result label with face and speaker info
        face_label = self.detected_face_label
        voice_label = speaker

        if face_label == voice_label:
            self.result_label.config(text=f"Match Found! Face: {face_label}, Speaker: {voice_label}", fg="green")
        else:
            self.result_label.config(text=f"No Match! Face: {face_label}, Speaker: {voice_label}", fg="red")


if __name__ == "__main__":
    from PIL import Image, ImageTk  # To handle image conversion for tkinter
    root = tk.Tk()
    app = Application(root)
    root.mainloop()
