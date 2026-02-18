import cv2
import numpy as np
import mediapipe as mp
from collections import deque

class EyeTracker:
    def __init__(self):
        """Initialize MediaPipe Face Mesh for eye tracking"""
        # Use the correct MediaPipe API
        mp_face_mesh = mp.solutions.face_mesh if hasattr(mp, 'solutions') else None
        
        if mp_face_mesh is None:
            # Fallback: Use simple eye tracker instead
            print("⚠️ MediaPipe solutions not available. Using simplified eye tracking.")
            from .simple_eye_tracker import SimpleEyeTracker
            self.simple_tracker = SimpleEyeTracker()
            self.face_mesh = None
            self.use_mediapipe = False
        else:
            self.simple_tracker = None
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,  # Enable iris landmarks
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.use_mediapipe = True
        
        # Iris landmarks indices (MediaPipe provides 468 face landmarks + 10 iris landmarks)
        self.LEFT_IRIS = [469, 470, 471, 472]
        self.RIGHT_IRIS = [474, 475, 476, 477]
        
        # Eye landmarks for calculating eye center
        self.LEFT_EYE = [33, 133, 160, 159, 158, 144, 145, 153]
        self.RIGHT_EYE = [362, 263, 387, 386, 385, 373, 374, 380]
        
        # Store gaze history for smoothing
        self.gaze_history = deque(maxlen=30)
        
    def calculate_gaze_ratio(self, eye_points, iris_center, frame_width):
        """Calculate where the iris is positioned relative to eye corners"""
        left_point = eye_points[0]
        right_point = eye_points[4]
        
        # Calculate eye width
        eye_width = np.linalg.norm(np.array(right_point) - np.array(left_point))
        
        # Calculate iris position relative to left corner
        iris_distance_from_left = np.linalg.norm(np.array(iris_center) - np.array(left_point))
        
        # Ratio: 0.5 = center, <0.3 = looking left, >0.7 = looking right
        if eye_width > 0:
            return iris_distance_from_left / eye_width
        return 0.5
    
    def get_iris_position(self, landmarks, iris_indices, frame_shape):
        """Get the center position of the iris"""
        h, w = frame_shape[:2]
        
        iris_points = []
        for idx in iris_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                x = int(lm.x * w)
                y = int(lm.y * h)
                iris_points.append([x, y])
        
        if len(iris_points) > 0:
            # Calculate center of iris
            iris_center = np.mean(iris_points, axis=0)
            return iris_center
        return None
    
    def get_eye_points(self, landmarks, eye_indices, frame_shape):
        """Get eye corner points"""
        h, w = frame_shape[:2]
        
        eye_points = []
        for idx in eye_indices:
            lm = landmarks[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            eye_points.append([x, y])
        
        return eye_points
    
    def classify_gaze(self, left_ratio, right_ratio):
        """Classify gaze direction based on iris ratios"""
        avg_ratio = (left_ratio + right_ratio) / 2
        
        # Looking at camera (centered)
        if 0.35 < avg_ratio < 0.65:
            return "center"
        # Looking left
        elif avg_ratio <= 0.35:
            return "left"
        # Looking right
        elif avg_ratio >= 0.65:
            return "right"
        
        return "center"
    
    def analyze_video(self, video_path):
        """
        Analyze eye movements in a video.
        Returns eye contact metrics and coaching feedback.
        """
        # Check if MediaPipe is available, use simple tracker as fallback
        if not self.use_mediapipe and self.simple_tracker is not None:
            print("⚠️ Using simplified eye tracking (MediaPipe not available)")
            return self.simple_tracker.analyze_video(video_path)
        elif not self.use_mediapipe:
            print("⚠️ Eye tracking unavailable - no tracker initialized")
            return {
                "error": "Eye tracking not available",
                "eye_contact_percentage": 0,
                "looking_away_count": 0,
                "gaze_stability": "N/A",
                "coaching_feedback": ["Eye tracking is currently unavailable."]
            }
        
        cap = cv2.VideoCapture(video_path)
        
        total_frames = 0
        frames_with_face = 0
        center_gaze_frames = 0
        looking_away_events = []
        
        current_gaze = "center"
        consecutive_away_frames = 0
        
        frame_count = 0
        skip_frames = 3  # Analyze every 3rd frame for efficiency
        
        print(f"👁️ Starting eye tracking analysis for: {video_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % skip_frames != 0:
                continue
            
            total_frames += 1
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                frames_with_face += 1
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Get iris positions
                left_iris_center = self.get_iris_position(landmarks, self.LEFT_IRIS, frame.shape)
                right_iris_center = self.get_iris_position(landmarks, self.RIGHT_IRIS, frame.shape)
                
                if left_iris_center is not None and right_iris_center is not None:
                    # Get eye points
                    left_eye_points = self.get_eye_points(landmarks, self.LEFT_EYE, frame.shape)
                    right_eye_points = self.get_eye_points(landmarks, self.RIGHT_EYE, frame.shape)
                    
                    # Calculate gaze ratios
                    left_ratio = self.calculate_gaze_ratio(left_eye_points, left_iris_center, frame.shape[1])
                    right_ratio = self.calculate_gaze_ratio(right_eye_points, right_iris_center, frame.shape[1])
                    
                    # Classify gaze
                    gaze_direction = self.classify_gaze(left_ratio, right_ratio)
                    self.gaze_history.append(gaze_direction)
                    
                    if gaze_direction == "center":
                        center_gaze_frames += 1
                        consecutive_away_frames = 0
                    else:
                        consecutive_away_frames += 1
                        
                        # Detect "looking away" event (looked away for 5+ consecutive frames)
                        if consecutive_away_frames == 5:
                            looking_away_events.append(frame_count)
                            current_gaze = gaze_direction
            
            if total_frames % 50 == 0:
                print(f"   ...Processed {total_frames} frames for eye tracking...")
        
        cap.release()
        print(f"✅ Eye tracking complete. Analyzed {frames_with_face} frames with detected faces.")
        
        if frames_with_face == 0:
            return {
                "error": "No faces detected for eye tracking",
                "eye_contact_percentage": 0,
                "looking_away_count": 0,
                "gaze_stability": "Unknown",
                "coaching_feedback": ["Could not detect eyes in the video. Ensure your face is clearly visible."]
            }
        
        # Calculate metrics
        eye_contact_percentage = round((center_gaze_frames / frames_with_face) * 100, 1)
        looking_away_count = len(looking_away_events)
        
        # Determine gaze stability
        if eye_contact_percentage >= 70:
            gaze_stability = "Excellent"
        elif eye_contact_percentage >= 50:
            gaze_stability = "Good"
        elif eye_contact_percentage >= 30:
            gaze_stability = "Fair"
        else:
            gaze_stability = "Needs Improvement"
        
        # Generate coaching feedback
        feedback = self.generate_coaching_feedback(
            eye_contact_percentage, 
            looking_away_count, 
            gaze_stability
        )
        
        return {
            "eye_contact_percentage": eye_contact_percentage,
            "looking_away_count": looking_away_count,
            "gaze_stability": gaze_stability,
            "coaching_feedback": feedback
        }
    
    def generate_coaching_feedback(self, eye_contact_pct, looking_away_count, stability):
        """Generate personalized coaching tips based on eye tracking metrics"""
        feedback = []
        
        # Feedback based on eye contact percentage
        if eye_contact_pct >= 70:
            feedback.append(f"Excellent eye contact! You maintained {eye_contact_pct}% eye contact.")
        elif eye_contact_pct >= 50:
            feedback.append(f"Good eye contact at {eye_contact_pct}%. Try to increase it to 70%+ for stronger engagement.")
        elif eye_contact_pct >= 30:
            feedback.append(f"Eye contact was {eye_contact_pct}%. Practice maintaining eye contact for longer periods.")
        else:
            feedback.append(f"Low eye contact detected ({eye_contact_pct}%). Focus on looking at the camera more consistently.")
        
        # Feedback based on looking away events
        if looking_away_count == 0:
            feedback.append("Great focus! You didn't look away during the response.")
        elif looking_away_count <= 2:
            feedback.append(f"You looked away {looking_away_count} time(s). This is acceptable, but try to minimize it.")
        else:
            feedback.append(f"You looked away {looking_away_count} times. Work on maintaining steady eye contact to appear more confident.")
        
        # General tip
        if eye_contact_pct < 60:
            feedback.append("💡 Tip: Imagine you're talking to a friend. Look directly at the camera lens as if making eye contact.")
        
        return feedback
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
