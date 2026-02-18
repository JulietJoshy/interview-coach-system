import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque
import os
from .eye_tracker import EyeTracker
from .drowsiness_detector import DrowsinessDetector

# Emotion Labels (must match training order)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class EmotionAnalyzer:
    def __init__(self, model_path='emotion_cnn_model.h5'):
        self.model = None
        self.model_path = model_path
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_history = deque(maxlen=30) # Store last 30 frames for smoothing
        self.eye_tracker = EyeTracker()  # Initialize eye tracker
        self.drowsiness_detector = DrowsinessDetector()  # Initialize drowsiness detector

    def load_saved_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                print("✅ Emotion Model Loaded Successfully")
                return True
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                return False
        else:
            print(f"⚠️ Model file not found at {self.model_path}. Please run train.py first.")
            return False

    def preprocess_face(self, face_img):
        # Convert to grayscale if not already
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Resize to 48x48
        face_img = cv2.resize(face_img, (48, 48))
        
        # Normalize
        face_img = face_img / 255.0
        
        # Reshape for model (1, 48, 48, 1)
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)
        
        return face_img

    def analyze_video(self, video_path):
        """
        Analyzes a video file for facial expressions.
        Returns a summary of emotions and coaching tips.
        """
        if self.model is None:
            if not self.load_saved_model():
                return {"error": "Emotion model not loaded. Please train the model first."}

        cap = cv2.VideoCapture(video_path)
        emotion_counts = {emotion: 0 for emotion in EMOTIONS}
        total_frames = 0
        
        frame_count = 0
        skip_frames = 12  # Analyze every 12th frame for faster processing
        max_analyzed_frames = 100  # Stop after analyzing 100 frames (enough for accurate summary)
        
        print(f"🎬 Starting analysis for: {video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Early stopping if we've analyzed enough frames
            if total_frames >= max_analyzed_frames:
                print(f"✅ Reached {max_analyzed_frames} analyzed frames. Stopping early for efficiency.")
                break
            
            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                 # Only take the largest face
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                (x, y, w, h) = faces[0]

                roi_gray = gray[y:y+h, x:x+w]
                processed_face = self.preprocess_face(roi_gray)
                
                prediction = self.model.predict(processed_face, verbose=0)
                max_index = int(np.argmax(prediction))
                predicted_emotion = EMOTIONS[max_index]
                
                emotion_counts[predicted_emotion] += 1
                total_frames += 1
            
            if frame_count % 30 == 0:
                print(f"   ...Processed {frame_count} frames...")

        cap.release()
        print(f"✅ Analysis Complete. Analyzed {total_frames} face frames.")
        
        if total_frames == 0:
            return {"message": "No faces detected in video."}
            
        # Calculate percentages
        emotion_summary = {k: round((v / total_frames) * 100, 2) for k, v in emotion_counts.items()}
        dominant_emotion = max(emotion_summary, key=emotion_summary.get)
        
        # Coaching Logic
        feedback = self.generate_feedback(dominant_emotion, emotion_summary)
        
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_breakdown": emotion_summary,
            "coaching_feedback": feedback
        }

    def generate_feedback(self, dominant, summary):
        feedback = []
        
        if dominant in ['Fear', 'Sad', 'Angry']:
            feedback.append(f"We detected a lot of {dominant} ({summary[dominant]}%). Try to relax and smile more.")
            
        if summary.get('Happy', 0) < 10:
            feedback.append("You didn't smile much. A warm smile helps build rapport.")
            
        if summary.get('Neutral', 0) > 80:
            feedback.append("Your expression was mostly static. Try to be more expressive to engage the interviewer.")
            
        if not feedback:
            feedback.append("Great job! Your facial expressions looked confident and positive.")
            
        return feedback
    
    def analyze_video_with_eye_tracking(self, video_path):
        """
        Comprehensive analysis: emotions + eye tracking + drowsiness.
        Returns combined results.
        """
        print("🎯 Running comprehensive analysis (Emotions + Eye Tracking + Drowsiness)...")
        
        # Run emotion analysis
        emotion_results = self.analyze_video(video_path)
        
        # Run eye tracking analysis
        eye_tracking_results = self.eye_tracker.analyze_video(video_path)
        
        # Run drowsiness analysis
        drowsiness_results = self.drowsiness_detector.analyze_video(video_path)
        
        # Merge results
        return {
            "emotion_analysis": emotion_results,
            "eye_tracking": eye_tracking_results,
            "drowsiness": drowsiness_results
        }
