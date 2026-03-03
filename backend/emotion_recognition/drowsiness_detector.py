import cv2
import numpy as np
from scipy.spatial import distance as dist
import os

class DrowsinessDetector:
    """
    Drowsiness detection using Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).
    Uses MediaPipe Face Mesh as primary, with dlib and OpenCV Haar cascades as fallbacks.
    """

    # EAR and MAR thresholds
    EAR_THRESHOLD = 0.25       # Below this = eyes closing (default, can be calibrated)
    MAR_THRESHOLD = 0.40       # Above this = yawning
    CONSECUTIVE_FRAMES = 3     # Frames below threshold to count as blink/drowsy
    YAWN_CONSECUTIVE = 2       # Frames above MAR threshold to count as yawn

    # MediaPipe landmark indices for EAR/MAR calculation
    MP_LEFT_EYE = [33, 160, 158, 133, 153, 144]   # p1-p6
    MP_RIGHT_EYE = [362, 385, 387, 263, 373, 380]  # p1-p6
    MP_INNER_MOUTH = [78, 81, 13, 311, 308, 402, 14, 82]  # 8 inner lip points

    def __init__(self):
        """Initialize drowsiness detector with MediaPipe, dlib, or fallback to OpenCV"""
        self.use_dlib = False
        self.use_mediapipe = False
        self.detector = None
        self.predictor = None

        # MediaPipe Face Mesh as primary detector
        try:
            import mediapipe as mp_lib
            mp_face_mesh = mp_lib.solutions.face_mesh if hasattr(mp_lib, 'solutions') else None
            if mp_face_mesh is not None:
                self.mp_face_mesh = mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.use_mediapipe = True
                print("✅ Drowsiness Detector: Using MediaPipe Face Mesh (primary)")
        except ImportError:
            print("⚠️ MediaPipe not available for drowsiness detection.")

        # dlib as secondary
        if not self.use_mediapipe:
            try:
                import dlib
                model_path = self._find_shape_predictor()
                if model_path:
                    self.detector = dlib.get_frontal_face_detector()
                    self.predictor = dlib.shape_predictor(model_path)
                    self.use_dlib = True
                    print("✅ Drowsiness Detector: Using dlib with 68-landmark predictor")
                else:
                    print("⚠️ dlib shape predictor not found. Downloading...")
                    self._download_shape_predictor()
                    model_path = self._find_shape_predictor()
                    if model_path:
                        self.detector = dlib.get_frontal_face_detector()
                        self.predictor = dlib.shape_predictor(model_path)
                        self.use_dlib = True
                        print("✅ Drowsiness Detector: dlib predictor downloaded and loaded")
                    else:
                        print("⚠️ Could not load dlib predictor. Using OpenCV fallback.")
            except ImportError:
                print("⚠️ dlib not installed. Using OpenCV fallback for drowsiness detection.")

        # OpenCV fallback detectors
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        # Mouth cascade for yawn detection
        mouth_cascade_path = cv2.data.haarcascades + 'haarcascade_smile.xml'
        self.mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)
        print(f"   Mouth cascade loaded: {not self.mouth_cascade.empty()}")

        # Per-user EAR calibration state
        self._calibration_ears = []
        self._calibrated_threshold = None
        self._calibration_frames = 30

    def reset(self):
        """Clear calibration state between videos"""
        self._calibration_ears = []
        self._calibrated_threshold = None

    @property
    def ear_threshold(self):
        """Returns calibrated EAR threshold or default"""
        if self._calibrated_threshold is not None:
            return self._calibrated_threshold
        return self.EAR_THRESHOLD

    def _calibrate_ear(self, ear_value):
        """Collect EAR values from first N frames for per-user calibration"""
        if len(self._calibration_ears) < self._calibration_frames:
            self._calibration_ears.append(ear_value)
            if len(self._calibration_ears) == self._calibration_frames:
                baseline = np.mean(self._calibration_ears)
                self._calibrated_threshold = baseline * 0.80
                print(f"   EAR calibrated: baseline={baseline:.3f}, threshold={self._calibrated_threshold:.3f}")

    def _find_shape_predictor(self):
        """Find the dlib shape predictor model file"""
        possible_paths = [
            os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'),
            os.path.join(os.path.dirname(__file__), '..', 'shape_predictor_68_face_landmarks.dat'),
            os.path.join(os.path.dirname(__file__), '..', 'data', 'shape_predictor_68_face_landmarks.dat'),
            'shape_predictor_68_face_landmarks.dat',
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    def _download_shape_predictor(self):
        """Download the dlib shape predictor model"""
        import urllib.request
        import bz2

        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        bz2_path = os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat.bz2')
        dat_path = os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat')

        try:
            print("📥 Downloading shape predictor model (~100MB)...")
            urllib.request.urlretrieve(url, bz2_path)

            print("📦 Extracting model...")
            with bz2.BZ2File(bz2_path) as fr, open(dat_path, 'wb') as fw:
                fw.write(fr.read())

            # Clean up bz2 file
            os.remove(bz2_path)
            print("✅ Shape predictor model ready!")
        except Exception as e:
            print(f"❌ Failed to download shape predictor: {e}")

    def _eye_aspect_ratio(self, eye_points):
        """
        Calculate Eye Aspect Ratio (EAR).

        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

        When eyes are open: EAR ≈ 0.25-0.30
        When eyes are closed: EAR ≈ 0.05
        """
        # Vertical distances
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        # Horizontal distance
        C = dist.euclidean(eye_points[0], eye_points[3])

        if C == 0:
            return 0.0

        ear = (A + B) / (2.0 * C)
        return ear

    def _mouth_aspect_ratio(self, mouth_points):
        """
        Calculate Mouth Aspect Ratio (MAR) for yawn detection.

        MAR = (|p2-p8| + |p3-p7| + |p4-p6|) / (3 * |p1-p5|)

        When mouth is closed: MAR ≈ 0.1-0.2
        When yawning: MAR > 0.6
        """
        # Vertical distances (inner lip points)
        A = dist.euclidean(mouth_points[1], mouth_points[7])
        B = dist.euclidean(mouth_points[2], mouth_points[6])
        C = dist.euclidean(mouth_points[3], mouth_points[5])
        # Horizontal distance
        D = dist.euclidean(mouth_points[0], mouth_points[4])

        if D == 0:
            return 0.0

        mar = (A + B + C) / (3.0 * D)
        return mar

    def _get_landmarks_points(self, shape, indices):
        """Extract landmark points as numpy array"""
        points = []
        for i in indices:
            points.append([shape.part(i).x, shape.part(i).y])
        return np.array(points)

    def _get_mp_landmark_points(self, landmarks, indices, frame_shape):
        """Extract MediaPipe landmark points as numpy array"""
        h, w = frame_shape[:2]
        points = []
        for idx in indices:
            lm = landmarks[idx]
            points.append([lm.x * w, lm.y * h])
        return np.array(points)

    def process_frame(self, gray, rgb_frame, landmarks, frame_shape):
        """
        Single-frame API for the unified inference pipeline.
        Accepts landmarks from eye_tracker to avoid duplicate face detection.
        Returns dict with ear, mar, eyes_closed, yawning or None.
        """
        if landmarks is not None:
            # Use provided MediaPipe landmarks
            left_eye = self._get_mp_landmark_points(landmarks, self.MP_LEFT_EYE, frame_shape)
            right_eye = self._get_mp_landmark_points(landmarks, self.MP_RIGHT_EYE, frame_shape)
            left_ear = self._eye_aspect_ratio(left_eye)
            right_ear = self._eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # Calibrate
            self._calibrate_ear(avg_ear)

            # Mouth
            inner_mouth = self._get_mp_landmark_points(landmarks, self.MP_INNER_MOUTH, frame_shape)
            mar = self._mouth_aspect_ratio(inner_mouth)

            eyes_closed = avg_ear < self.ear_threshold
            yawning = mar > self.MAR_THRESHOLD

            return {
                "ear": avg_ear,
                "mar": mar,
                "eyes_closed": eyes_closed,
                "yawning": yawning
            }

        # Fallback: use MediaPipe internally
        if self.use_mediapipe:
            results = self.mp_face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                mp_landmarks = results.multi_face_landmarks[0].landmark
                return self.process_frame(gray, rgb_frame, mp_landmarks, frame_shape)

        # Fallback: dlib
        if self.use_dlib and self.detector and self.predictor:
            faces = self.detector(gray, 0)
            if len(faces) > 0:
                face = faces[0]
                shape = self.predictor(gray, face)
                left_eye = self._get_landmarks_points(shape, range(36, 42))
                right_eye = self._get_landmarks_points(shape, range(42, 48))
                left_ear = self._eye_aspect_ratio(left_eye)
                right_ear = self._eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                self._calibrate_ear(avg_ear)

                inner_mouth = self._get_landmarks_points(shape, range(60, 68))
                mar = self._mouth_aspect_ratio(inner_mouth)

                return {
                    "ear": avg_ear,
                    "mar": mar,
                    "eyes_closed": avg_ear < self.ear_threshold,
                    "yawning": mar > self.MAR_THRESHOLD
                }

        return None

    def analyze_video(self, video_path):
        """
        Analyze video for drowsiness indicators.
        Returns alertness metrics and coaching feedback.
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return self._error_result("Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_video_frames / fps if fps > 0 else 0

        # Tracking variables
        total_frames_analyzed = 0
        frames_with_face = 0

        # EAR tracking
        ear_values = []
        frames_eyes_closed = 0
        total_eye_frames = 0

        # Blink tracking
        blink_count = 0
        consecutive_closed = 0
        was_closed = False

        # Yawn tracking
        yawn_count = 0
        consecutive_yawn = 0
        mar_values = []

        frame_count = 0
        skip_frames = 3  # Analyze every 3rd frame

        print(f"😴 Starting drowsiness analysis for: {video_path}")
        print(f"   Video: {total_video_frames} frames, {video_duration:.1f}s, {fps:.0f} FPS")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            total_frames_analyzed += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.use_dlib and self.detector and self.predictor:
                # === dlib-based analysis (more accurate) ===
                faces = self.detector(gray, 0)

                if len(faces) > 0:
                    frames_with_face += 1
                    face = faces[0]
                    shape = self.predictor(gray, face)

                    # Eye landmarks: left eye (36-41), right eye (42-47)
                    left_eye = self._get_landmarks_points(shape, range(36, 42))
                    right_eye = self._get_landmarks_points(shape, range(42, 48))

                    left_ear = self._eye_aspect_ratio(left_eye)
                    right_ear = self._eye_aspect_ratio(right_eye)
                    avg_ear = (left_ear + right_ear) / 2.0
                    ear_values.append(avg_ear)
                    total_eye_frames += 1

                    # Check if eyes are closing
                    if avg_ear < self.EAR_THRESHOLD:
                        frames_eyes_closed += 1
                        consecutive_closed += 1

                        if consecutive_closed >= self.CONSECUTIVE_FRAMES and not was_closed:
                            blink_count += 1
                            was_closed = True
                    else:
                        consecutive_closed = 0
                        was_closed = False

                    # Mouth landmarks: inner lip (60-67)
                    inner_mouth = self._get_landmarks_points(shape, range(60, 68))
                    mar = self._mouth_aspect_ratio(inner_mouth)
                    mar_values.append(mar)

                    # Check for yawning
                    if mar > self.MAR_THRESHOLD:
                        consecutive_yawn += 1
                        if consecutive_yawn == self.YAWN_CONSECUTIVE:
                            yawn_count += 1
                    else:
                        consecutive_yawn = 0
            else:
                # === OpenCV fallback (with mouth/yawn detection) ===
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                if len(faces) > 0:
                    frames_with_face += 1
                    (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]

                    roi_gray = gray[y:y+h, x:x+w]

                    # --- Eye detection ---
                    # Only look in top 60% of face for eyes
                    eye_region = roi_gray[0:int(h*0.6), :]
                    eyes = self.eye_cascade.detectMultiScale(eye_region, 1.1, 5)
                    total_eye_frames += 1

                    if len(eyes) < 2:
                        # Less than 2 eyes detected = possibly closed/drowsy
                        frames_eyes_closed += 1
                        consecutive_closed += 1

                        if consecutive_closed >= self.CONSECUTIVE_FRAMES and not was_closed:
                            blink_count += 1
                            was_closed = True
                    else:
                        consecutive_closed = 0
                        was_closed = False

                        # Estimate EAR from eye detection dimensions
                        for (ex, ey, ew, eh) in eyes[:2]:
                            ear_approx = eh / (ew + 0.001)
                            ear_values.append(ear_approx)

                    # --- Mouth/Yawn detection ---
                    # Look at bottom 50% of face for mouth
                    mouth_y_start = int(h * 0.5)
                    mouth_region = roi_gray[mouth_y_start:, :]

                    if mouth_region.size > 0:
                        # Detect mouth openings
                        mouths = self.mouth_cascade.detectMultiScale(
                            mouth_region,
                            scaleFactor=1.7,
                            minNeighbors=11,
                            minSize=(int(w*0.25), int(h*0.1))
                        )

                        if len(mouths) > 0:
                            # Get the largest mouth detection
                            (mx, my, mw, mh) = sorted(mouths, key=lambda m: m[2]*m[3], reverse=True)[0]

                            # Calculate mouth aspect ratio from bounding box
                            mar_approx = mh / (mw + 0.001)
                            mar_values.append(mar_approx)

                            # Yawn: mouth is open wide (height > 50% of width)
                            if mar_approx > self.MAR_THRESHOLD:
                                consecutive_yawn += 1
                                if consecutive_yawn == self.YAWN_CONSECUTIVE:
                                    yawn_count += 1
                                    print(f"   🥱 Yawn detected! (MAR: {mar_approx:.2f}) Total: {yawn_count}")
                            else:
                                consecutive_yawn = 0
                        else:
                            consecutive_yawn = 0

            if total_frames_analyzed % 50 == 0:
                print(f"   ...Processed {total_frames_analyzed} frames for drowsiness...")

        cap.release()
        print(f"✅ Drowsiness analysis complete. Analyzed {frames_with_face} frames with faces.")

        # Calculate metrics
        if frames_with_face == 0:
            return self._error_result("No faces detected for drowsiness analysis")

        # PERCLOS: Percentage of time eyes are closed
        perclos = round((frames_eyes_closed / total_eye_frames) * 100, 1) if total_eye_frames > 0 else 0

        # Average EAR
        avg_ear = round(np.mean(ear_values), 3) if ear_values else 0.25

        # Blink rate (blinks per minute)
        blink_rate = round((blink_count / video_duration) * 60, 1) if video_duration > 0 else 0

        # Calculate alertness score (0-100)
        alertness_score = self._calculate_alertness_score(perclos, yawn_count, blink_rate, avg_ear)

        # Determine drowsiness level
        drowsiness_level = self._get_drowsiness_level(alertness_score)

        # Generate coaching feedback
        feedback = self._generate_feedback(alertness_score, drowsiness_level, yawn_count, blink_rate, perclos)

        return {
            "alertness_score": alertness_score,
            "drowsiness_level": drowsiness_level,
            "yawn_count": yawn_count,
            "blink_rate": blink_rate,
            "perclos": perclos,
            "avg_ear": avg_ear,
            "coaching_feedback": feedback
        }

    def _calculate_alertness_score(self, perclos, yawn_count, blink_rate, avg_ear):
        """Calculate overall alertness score (0-100)"""
        score = 100

        # PERCLOS penalty (biggest indicator)
        if perclos > 40:
            score -= 40
        elif perclos > 25:
            score -= 25
        elif perclos > 15:
            score -= 15
        elif perclos > 8:
            score -= 5

        # Yawn penalty
        score -= min(yawn_count * 8, 25)

        # Blink rate penalty (normal: 12-20/min)
        if blink_rate > 25:
            score -= 15  # Excessive blinking = fatigue
        elif blink_rate > 20:
            score -= 5
        elif blink_rate < 8 and blink_rate > 0:
            score -= 5   # Too few blinks = staring/zoned out

        # EAR penalty (low average = droopy eyes)
        if avg_ear < 0.20:
            score -= 15
        elif avg_ear < 0.22:
            score -= 8

        return max(0, min(100, round(score)))

    def _get_drowsiness_level(self, alertness_score):
        """Determine drowsiness level from alertness score"""
        if alertness_score >= 80:
            return "Alert"
        elif alertness_score >= 60:
            return "Mild Drowsiness"
        elif alertness_score >= 40:
            return "Moderate Drowsiness"
        else:
            return "Severe Drowsiness"

    def _generate_feedback(self, score, level, yawn_count, blink_rate, perclos):
        """Generate personalized coaching feedback"""
        feedback = []

        # Overall assessment
        if score >= 80:
            feedback.append(f"✅ You appeared alert and energetic! Alertness score: {score}/100.")
        elif score >= 60:
            feedback.append(f"⚠️ Mild signs of drowsiness detected. Alertness score: {score}/100.")
        elif score >= 40:
            feedback.append(f"⚠️ Moderate drowsiness detected. Alertness score: {score}/100. This could impact your interview impression.")
        else:
            feedback.append(f"🚨 Signs of significant drowsiness detected. Alertness score: {score}/100. Consider resting before interviews.")

        # Yawn feedback
        if yawn_count == 0:
            feedback.append("No yawning detected — great energy!")
        elif yawn_count <= 2:
            feedback.append(f"You yawned {yawn_count} time(s). This could signal tiredness to interviewers.")
        else:
            feedback.append(f"You yawned {yawn_count} times. Frequent yawning may give a negative impression. Ensure you're well-rested.")

        # Blink rate feedback
        if 12 <= blink_rate <= 20:
            feedback.append(f"Blink rate is normal ({blink_rate}/min).")
        elif blink_rate > 20:
            feedback.append(f"High blink rate ({blink_rate}/min) may indicate fatigue or stress.")
        elif blink_rate > 0:
            feedback.append(f"Low blink rate ({blink_rate}/min). Try to blink naturally to avoid appearing tense.")

        # PERCLOS feedback
        if perclos > 20:
            feedback.append(f"Your eyes were closed {perclos}% of the time. Keep your eyes open wide to appear engaged.")

        # General tips for low scores
        if score < 70:
            feedback.append("💡 Tip: Get adequate sleep, stay hydrated, and do light stretches before your interview.")

        return feedback

    def _error_result(self, message):
        """Return error result with default values"""
        return {
            "error": message,
            "alertness_score": 0,
            "drowsiness_level": "Unknown",
            "yawn_count": 0,
            "blink_rate": 0,
            "perclos": 0,
            "avg_ear": 0,
            "coaching_feedback": [f"⚠️ {message}. Ensure good lighting and face visibility."]
        }
