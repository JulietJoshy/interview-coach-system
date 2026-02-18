import cv2
import numpy as np

class SimpleEyeTracker:
    """Simple eye tracking without MediaPipe - estimates based on face position"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
    def analyze_video(self, video_path):
        """
        Simplified eye tracking using basic face/eye detection.
        Returns estimated eye contact metrics.
        """
        cap = cv2.VideoCapture(video_path)
        
        total_frames = 0
        frames_with_face = 0
        frames_with_both_eyes = 0
        face_center_history = []
        
        frame_count = 0
        skip_frames = 5  # Analyze every 5th frame
        
        frame_width = None
        
        print(f"👁️ Starting simplified eye tracking for: {video_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_width is None:
                frame_width = frame.shape[1]
            
            frame_count += 1
            if frame_count % skip_frames != 0:
                continue
            
            total_frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect face
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                frames_with_face += 1
                # Get largest face
                (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                
                # Calculate face center as percentage of frame width
                face_center_x = (x + w/2) / frame_width
                face_center_history.append(face_center_x)
                
                # Detect eyes within face region
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
                
                if len(eyes) >= 2:
                    frames_with_both_eyes += 1
        
        cap.release()
        print(f"✅ Eye tracking complete. Analyzed {frames_with_face} frames with faces.")
        
        if frames_with_face == 0:
            return {
                "eye_contact_percentage": 0,
                "looking_away_count": 0,
                "gaze_stability": "Unknown",
                "coaching_feedback": ["Could not detect face in video. Ensure good lighting and face centered."]
            }
        
        # Estimate metrics
        # Eye contact % based on how often both eyes are visible
        eye_contact_percentage = round((frames_with_both_eyes / frames_with_face) * 100, 1)
        
        # Estimate looking away events based on face center position variance
        looking_away_count = self._estimate_looking_away(face_center_history)
        
        # Determine gaze stability
        if eye_contact_percentage == 0:
            gaze_stability = "Poor Detection"
        elif eye_contact_percentage >= 70:
            gaze_stability = "Excellent"
        elif eye_contact_percentage >= 50:
            gaze_stability = "Good"
        elif eye_contact_percentage >= 30:
            gaze_stability = "Fair"
        else:
            gaze_stability = "Needs Improvement"
        
        feedback = self._generate_feedback(eye_contact_percentage, looking_away_count)
        
        return {
            "eye_contact_percentage": eye_contact_percentage,
            "looking_away_count": looking_away_count,
            "gaze_stability": gaze_stability,
            "coaching_feedback": feedback
        }
    
    def _estimate_looking_away(self, face_center_history):
        """Estimate looking away events from face position variance"""
        if len(face_center_history) < 5:
            return 0
        
        # Count how many times face center moves significantly
        looking_away = 0
        for i in range(1, len(face_center_history)):
            # If face center moves more than 15% from center (0.5)
            if abs(face_center_history[i] - 0.5) > 0.15:
                if abs(face_center_history[i-1] - 0.5) <= 0.15:
                    # Transitioned from center to away
                    looking_away += 1
        
        return looking_away
    
    def _generate_feedback(self, eye_contact_pct, looking_away_count):
        """Generate coaching feedback"""
        feedback = []
        
        # Special case: If eye contact is 0%, detection failed
        if eye_contact_pct == 0:
            feedback.append("⚠️ Could not detect eyes in the video.")
            feedback.append("Make sure your face is well-lit and clearly visible to the camera.")
            feedback.append("💡 Tip: Position yourself in good lighting with your face centered in frame.")
            return feedback
        
        # Normal feedback for successful detection
        if eye_contact_pct >= 70:
            feedback.append(f"Good job! Maintained face visibility at {eye_contact_pct}%.")
        elif eye_contact_pct >= 50:
            feedback.append(f"Decent presence at {eye_contact_pct}%. Try to stay more centered.")
        elif eye_contact_pct >= 30:
            feedback.append(f"Some visibility at {eye_contact_pct}%. Work on keeping your face in frame more consistently.")
        else:
            feedback.append(f"Low visibility at {eye_contact_pct}%. Ensure your face stays clearly in frame.")
        
        # Only provide movement feedback if we had decent detection
        if eye_contact_pct >= 30:
            if looking_away_count <= 2:
                feedback.append(f"Good stability - minimal head movement detected.")
            else:
                feedback.append(f"You moved your head {looking_away_count} times. Try to stay more still.")
        
        feedback.append("💡 Tip: Keep your face centered in the camera for best impression.")
        
        return feedback
