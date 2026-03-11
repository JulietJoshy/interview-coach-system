import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque
import os
import urllib.request
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

        # DNN face detector (loaded lazily)
        self._dnn_net = None
        self._dnn_loaded = False

        # CLAHE for preprocessing
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _ensure_dnn_detector(self):
        """Load OpenCV DNN face detector (res10_300x300). Downloads on first use."""
        if self._dnn_loaded:
            return self._dnn_net is not None

        self._dnn_loaded = True
        base_dir = os.path.dirname(os.path.abspath(__file__))
        prototxt_path = os.path.join(base_dir, 'deploy.prototxt')
        caffemodel_path = os.path.join(base_dir, 'res10_300x300_ssd_iter_140000.caffemodel')

        # Download if not present
        if not os.path.exists(prototxt_path) or not os.path.exists(caffemodel_path):
            try:
                print("📥 Downloading DNN face detector model...")
                prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
                caffemodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

                if not os.path.exists(prototxt_path):
                    urllib.request.urlretrieve(prototxt_url, prototxt_path)
                if not os.path.exists(caffemodel_path):
                    urllib.request.urlretrieve(caffemodel_url, caffemodel_path)
                print("✅ DNN face detector downloaded.")
            except Exception as e:
                print(f"⚠️ Could not download DNN face detector: {e}. Using Haar cascade fallback.")
                return False

        try:
            self._dnn_net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
            print("✅ DNN face detector loaded.")
            return True
        except Exception as e:
            print(f"⚠️ Could not load DNN face detector: {e}. Using Haar cascade fallback.")
            return False

    def _detect_face_dnn(self, frame, confidence_threshold=0.5):
        """Detect face using DNN. Returns (x, y, w, h) or None."""
        if self._dnn_net is None:
            return None

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                      (300, 300), (104.0, 177.0, 123.0))
        self._dnn_net.setInput(blob)
        detections = self._dnn_net.forward()

        best_detection = None
        best_confidence = confidence_threshold

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > best_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                # Clamp to frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                if x2 > x1 and y2 > y1:
                    best_confidence = confidence
                    best_detection = (x1, y1, x2 - x1, y2 - y1)

        return best_detection

    def _detect_face(self, frame, gray):
        """Detect face using DNN first, Haar cascade fallback. Returns (x, y, w, h) or None."""
        # Try DNN
        if self._dnn_net is not None:
            result = self._detect_face_dnn(frame)
            if result is not None:
                return result

        # Haar cascade fallback
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            return tuple(faces[0])

        return None

    def load_saved_model(self):
        """Load model with backward compatibility: full model → v2 weights → v1 weights"""
        if os.path.exists(self.model_path):
            # Try 1: Load full saved model
            try:
                self.model = load_model(self.model_path)
                print("✅ Emotion Model Loaded Successfully")
                return True
            except Exception as e:
                print(f"⚠️ Full model load failed: {e}. Trying architecture + weights...")

            # Try 2: V2 architecture + weights
            try:
                from .model import create_emotion_model_v2
                self.model = create_emotion_model_v2(input_shape=(48, 48, 1), num_classes=7)
                self.model.load_weights(self.model_path)
                print("✅ Emotion Model Loaded (V2 architecture + weights)")
                return True
            except Exception as e:
                print(f"⚠️ V2 architecture load failed: {e}. Trying V1...")

            # Try 3: V1 architecture + weights
            try:
                from .model import create_emotion_model
                self.model = create_emotion_model(input_shape=(48, 48, 1), num_classes=7)
                self.model.load_weights(self.model_path)
                print("✅ Emotion Model Loaded (V1 architecture + weights)")
                return True
            except Exception as e:
                print(f"❌ Failed to load model with any method: {e}")
                return False
        else:
            print(f"⚠️ Model file not found at {self.model_path}. Please run train.py first.")
            return False

    def preprocess_face(self, face_img):
        # Convert to grayscale if not already
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for better contrast
        face_img = self._clahe.apply(face_img)

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

    def analyze_video_single_pass(self, video_path):
        """
        Single-pass video analysis: read frames once, share across all analyzers.
        Uses DNN face detection, CLAHE preprocessing, confidence thresholding,
        and temporal smoothing for improved accuracy.
        """
        print("🎯 Running single-pass analysis pipeline...")

        if self.model is None:
            if not self.load_saved_model():
                return None  # Will trigger fallback

        # Ensure DNN detector is ready
        self._ensure_dnn_detector()

        # Reset per-video state
        self.eye_tracker.reset()
        self.drowsiness_detector.reset()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_video_frames / fps if fps > 0 else 0

        # Emotion tracking — probability arrays for weighted averaging
        emotion_prob_arrays = []
        emotion_weights = []

        # Eye tracking
        eye_center_frames = 0
        eye_frames_with_face = 0
        eye_looking_away_events = []
        eye_consecutive_away = 0

        # Drowsiness tracking
        ear_values = []
        frames_eyes_closed = 0
        total_eye_frames = 0
        blink_count = 0
        consecutive_closed = 0
        was_closed = False
        yawn_count = 0
        consecutive_yawn = 0
        mar_values = []

        frame_count = 0
        analyzed_frames = 0
        skip_frames = 6   # Analyze every 6th frame (2x more than old pipeline)
        max_frames = 200

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if analyzed_frames >= max_frames:
                print(f"✅ Reached {max_frames} analyzed frames. Stopping.")
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            analyzed_frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- Drowsiness Detection runs FIRST (it reliably detects faces) ---
            # Pass mp_landmarks=None so it uses its own FaceLandmarker to detect
            drowsy_result = self.drowsiness_detector.process_frame(
                gray, rgb_frame, None, frame.shape
            )
            shared_landmarks = None  # Will be populated after drowsiness detection

            if drowsy_result is not None:
                ear_values.append(drowsy_result["ear"])
                total_eye_frames += 1
                # Grab landmarks from drowsiness detector for eye tracking reuse
                shared_landmarks = drowsy_result.get("landmarks")

                if drowsy_result["eyes_closed"]:
                    frames_eyes_closed += 1
                    consecutive_closed += 1
                    if consecutive_closed >= self.drowsiness_detector.CONSECUTIVE_FRAMES and not was_closed:
                        blink_count += 1
                        was_closed = True
                else:
                    consecutive_closed = 0
                    was_closed = False

                mar_values.append(drowsy_result["mar"])
                if drowsy_result["yawning"]:
                    consecutive_yawn += 1
                    if consecutive_yawn == self.drowsiness_detector.YAWN_CONSECUTIVE:
                        yawn_count += 1
                else:
                    consecutive_yawn = 0

            # --- Eye Tracking: reuse landmarks from DrowsinessDetector ---
            if shared_landmarks is not None:
                eye_result = self.eye_tracker.analyze_gaze_from_landmarks(
                    shared_landmarks, frame.shape
                )
                if eye_result is not None:
                    eye_frames_with_face += 1
                    if eye_result["looking_at_camera"]:
                        eye_center_frames += 1
                        eye_consecutive_away = 0
                    else:
                        eye_consecutive_away += 1
                        if eye_consecutive_away == 3:  # Reduced: ~0.6s of looking away counts
                            eye_looking_away_events.append(frame_count)

            # --- Emotion Detection (shared face detection) ---
            face_rect = self._detect_face(frame, gray)
            if face_rect is not None and self.model is not None:
                (x, y, w, h) = face_rect
                roi_gray = gray[y:y+h, x:x+w]
                if roi_gray.size > 0:
                    processed_face = self.preprocess_face(roi_gray)
                    prediction = self.model.predict(processed_face, verbose=0)[0]

                    # Confidence thresholding
                    max_confidence = float(np.max(prediction))
                    if max_confidence > 0.4:
                        # Weight by recency (later frames weighted higher)
                        weight = 0.5 + 0.5 * (analyzed_frames / max_frames)
                        emotion_prob_arrays.append(prediction)
                        emotion_weights.append(weight)

            if analyzed_frames % 50 == 0:
                print(f"   ...Single-pass: processed {analyzed_frames} frames...")

        cap.release()
        print(f"✅ Single-pass analysis complete. {analyzed_frames} frames analyzed.")

        # --- Compute Results ---
        emotion_results = self._compute_emotion_results(emotion_prob_arrays, emotion_weights)
        eye_results = self._compute_eye_results(
            eye_center_frames, eye_frames_with_face, eye_looking_away_events
        )
        drowsiness_results = self._compute_drowsiness_results(
            ear_values, frames_eyes_closed, total_eye_frames,
            blink_count, yawn_count, mar_values, video_duration
        )

        return {
            "emotion_analysis": emotion_results,
            "eye_tracking": eye_results,
            "drowsiness": drowsiness_results
        }

    def _compute_emotion_results(self, prob_arrays, weights):
        """Compute emotion results from weighted probability arrays."""
        if not prob_arrays:
            return {"message": "No faces detected in video."}

        # Weighted average of probability arrays
        prob_arrays = np.array(prob_arrays)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        avg_probs = np.average(prob_arrays, axis=0, weights=weights)

        # Convert to percentages
        emotion_summary = {}
        for i, emotion in enumerate(EMOTIONS):
            emotion_summary[emotion] = round(float(avg_probs[i]) * 100, 2)

        dominant_emotion = max(emotion_summary, key=emotion_summary.get)
        feedback = self.generate_feedback(dominant_emotion, emotion_summary)

        return {
            "dominant_emotion": dominant_emotion,
            "emotion_breakdown": emotion_summary,
            "coaching_feedback": feedback
        }

    def _compute_eye_results(self, center_frames, frames_with_face, looking_away_events):
        """Compute eye tracking results using same thresholds as EyeTracker."""
        if frames_with_face == 0:
            return {
                "error": "No faces detected for eye tracking",
                "eye_contact_percentage": 0,
                "looking_away_count": 0,
                "gaze_stability": "Unknown",
                "coaching_feedback": ["Could not detect eyes in the video. Ensure your face is clearly visible."]
            }

        eye_contact_percentage = round((center_frames / frames_with_face) * 100, 1)
        looking_away_count = len(looking_away_events)

        if eye_contact_percentage >= 70:
            gaze_stability = "Excellent"
        elif eye_contact_percentage >= 50:
            gaze_stability = "Good"
        elif eye_contact_percentage >= 30:
            gaze_stability = "Fair"
        else:
            gaze_stability = "Needs Improvement"

        feedback = self.eye_tracker.generate_coaching_feedback(
            eye_contact_percentage, looking_away_count, gaze_stability
        )

        return {
            "eye_contact_percentage": eye_contact_percentage,
            "looking_away_count": looking_away_count,
            "gaze_stability": gaze_stability,
            "coaching_feedback": feedback
        }

    def _compute_drowsiness_results(self, ear_values, frames_eyes_closed,
                                     total_eye_frames, blink_count, yawn_count,
                                     mar_values, video_duration):
        """Compute drowsiness results using DrowsinessDetector's scoring methods."""
        if total_eye_frames == 0:
            return self.drowsiness_detector._error_result("No faces detected for drowsiness analysis")

        perclos = round((frames_eyes_closed / total_eye_frames) * 100, 1)
        avg_ear = round(np.mean(ear_values), 3) if ear_values else 0.25
        blink_rate = round((blink_count / video_duration) * 60, 1) if video_duration > 0 else 0

        alertness_score = self.drowsiness_detector._calculate_alertness_score(
            perclos, yawn_count, blink_rate, avg_ear
        )
        drowsiness_level = self.drowsiness_detector._get_drowsiness_level(alertness_score)
        feedback = self.drowsiness_detector._generate_feedback(
            alertness_score, drowsiness_level, yawn_count, blink_rate, perclos
        )

        return {
            "alertness_score": alertness_score,
            "drowsiness_level": drowsiness_level,
            "yawn_count": yawn_count,
            "blink_rate": blink_rate,
            "perclos": perclos,
            "avg_ear": avg_ear,
            "coaching_feedback": feedback
        }

    def analyze_video_with_eye_tracking(self, video_path):
        """
        Comprehensive analysis: emotions + eye tracking + drowsiness.
        Returns combined results. Uses single-pass pipeline with 3-pass fallback.
        """
        print("🎯 Running comprehensive analysis (Emotions + Eye Tracking + Drowsiness)...")

        # Try single-pass pipeline first
        try:
            result = self.analyze_video_single_pass(video_path)
            if result is not None:
                print("✅ Single-pass analysis succeeded.")
                return result
        except Exception as e:
            print(f"⚠️ Single-pass analysis failed: {e}. Falling back to 3-pass approach.")

        # Fall back to original 3-pass approach
        print("🔄 Using 3-pass fallback analysis...")
        emotion_results = self.analyze_video(video_path)
        eye_tracking_results = self.eye_tracker.analyze_video(video_path)
        drowsiness_results = self.drowsiness_detector.analyze_video(video_path)

        return {
            "emotion_analysis": emotion_results,
            "eye_tracking": eye_tracking_results,
            "drowsiness": drowsiness_results
        }
