import cv2
import numpy as np
import os
from collections import deque

# ─── API availability ─────────────────────────────────────────────────────────
_FACE_MESH_AVAILABLE = False
_TASKS_AVAILABLE     = False
_mp_module = None

try:
    import mediapipe as mp
    _mp_module = mp

    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
        _FACE_MESH_AVAILABLE = True
        print("✅ EyeTracker: mp.solutions.face_mesh available")

    try:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.vision import FaceLandmarkerOptions
        _TASKS_AVAILABLE = True
    except Exception:
        pass

except Exception as e:
    print(f"⚠️ EyeTracker: MediaPipe not available: {e}")


class EyeTracker:
    """
    Eye-contact tracker for interview coaching.

    PRIMARY signal  — nose centering between eye corners:
        When you face the camera your nose sits ~0.5 of the way between
        the outer eye corners.  Head turns shift the nose clearly off-centre.
        This gives a strong, reliable look-away signal that iris-ratio cannot.

    SECONDARY signal — iris-within-eye ratio (catches eye-only gaze shifts).

    Both signals are checked; either one failing means "looking away".
    """

    # ── Landmark indices (works for both FaceMesh and FaceLandmarker) ─────────
    LEFT_IRIS   = [469, 470, 471, 472]
    RIGHT_IRIS  = [474, 475, 476, 477]
    LEFT_EYE    = [33,  133, 160, 159, 158, 144, 145, 153]  # [0]=outer, [1]=inner
    RIGHT_EYE   = [263, 362, 387, 386, 385, 373, 374, 380]  # [0]=outer, [1]=inner
    NOSE_TIP    = 4    # actual nose tip (more forward than landmark 1)
    L_EYE_OUT   = 33   # left  eye outer corner (temple side)
    R_EYE_OUT   = 263  # right eye outer corner (temple side)

    # ── Thresholds ────────────────────────────────────────────────────────────
    # Nose centering: 0 = at left eye corner, 1 = at right eye corner
    # When facing camera nose is ~0.5; outside 0.35–0.65 → head turned
    NOSE_CENTER_MIN = 0.35
    NOSE_CENTER_MAX = 0.65
    # Iris ratio: 0.35–0.65 = looking at camera
    IRIS_RATIO_MIN  = 0.30
    IRIS_RATIO_MAX  = 0.70

    def __init__(self):
        self.face_mesh       = None
        self.face_landmarker = None
        self.use_face_mesh   = False
        self.use_tasks       = False

        self.gaze_history = deque(maxlen=30)
        self._gaze_window = deque(maxlen=3)

        # ── Classic FaceMesh (preferred — reference approach) ─────────────────
        if _FACE_MESH_AVAILABLE:
            try:
                self.face_mesh = _mp_module.solutions.face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,   # enables iris landmarks 469-477
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.use_face_mesh = True
                print("✅ EyeTracker: FaceMesh initialised")
            except Exception as e:
                print(f"⚠️ EyeTracker: FaceMesh init failed: {e}")

        # ── Tasks API fallback ────────────────────────────────────────────────
        # NOTE: Skipped intentionally. In the main pipeline, EyeTracker always
        # receives landmarks shared from DrowsinessDetector (which has its own
        # FaceLandmarker). Creating a second FaceLandmarker here causes a
        # mediapipe C-library conflict and crashes the backend on startup.
        # The Tasks API path in _detect_tasks() is only used by the standalone
        # analyze_video() fallback, which is rarely triggered.
        # To re-enable: set self.use_tasks = True and init self.face_landmarker.
        self.use_tasks = False
        self.face_landmarker = None

        if not self.use_face_mesh and not self.use_tasks:
            print("⚠️ EyeTracker: no MediaPipe API available (FaceMesh will be used if available)")

    @property
    def use_mediapipe(self):
        return self.use_face_mesh or self.use_tasks

    def reset(self):
        self.gaze_history.clear()
        self._gaze_window.clear()

    # =========================================================================
    # Geometry helpers
    # =========================================================================

    def _lm_px(self, lm, w, h):
        """Landmark → pixel coords."""
        return lm.x * w, lm.y * h

    def get_iris_position(self, landmarks, iris_indices, frame_shape):
        h, w = frame_shape[:2]
        pts = [[landmarks[i].x * w, landmarks[i].y * h]
               for i in iris_indices if i < len(landmarks)]
        return np.mean(pts, axis=0) if pts else None

    def get_eye_points(self, landmarks, eye_indices, frame_shape):
        h, w = frame_shape[:2]
        return [[landmarks[i].x * w, landmarks[i].y * h] for i in eye_indices]

    def calculate_iris_ratio(self, eye_points, iris_center):
        """
        Iris position as fraction of horizontal eye width.
        Uses index [0] (outer corner) and [1] (inner corner) — the true edges.
        Returns ~0.5 when iris is centred.
        """
        p0 = np.array(eye_points[0], float)
        p1 = np.array(eye_points[1], float)
        lp, rp = (p0, p1) if p0[0] < p1[0] else (p1, p0)   # left → right
        width = np.linalg.norm(rp - lp)
        if width == 0:
            return 0.5
        return float(np.linalg.norm(np.array(iris_center, float) - lp) / width)

    # =========================================================================
    # PRIMARY SIGNAL: nose centering
    # =========================================================================

    def get_nose_centering(self, landmarks, frame_shape):
        """
        Returns the nose-tip x-position normalised by the span between the two
        outer eye corners (0 = at left-screen corner, 1 = at right-screen corner).

        Expected value when facing camera: ~0.45–0.55
        Head turned to person's RIGHT  (nose shifts left on screen): value < 0.35
        Head turned to person's LEFT   (nose shifts right on screen): value > 0.65
        """
        h, w = frame_shape[:2]
        try:
            nose_x = landmarks[self.NOSE_TIP].x * w
            l_x    = landmarks[self.L_EYE_OUT].x * w
            r_x    = landmarks[self.R_EYE_OUT].x * w
            left_x, right_x = min(l_x, r_x), max(l_x, r_x)
            span = right_x - left_x
            if span < 10:
                return None
            return (nose_x - left_x) / span
        except (IndexError, AttributeError):
            return None

    # =========================================================================
    # Combined classifier
    # =========================================================================

    def _is_looking(self, landmarks, frame_shape):
        """
        True = looking at camera.
        Uses nose centering (primary) + iris ratio (secondary).
        """
        h, w = frame_shape[:2]

        # ── SIGNAL 1: nose centering (head-turn detector) ─────────────────────
        nose_pos = self.get_nose_centering(landmarks, frame_shape)
        if nose_pos is not None:
            if nose_pos < self.NOSE_CENTER_MIN or nose_pos > self.NOSE_CENTER_MAX:
                return False   # head clearly turned away

        # ── SIGNAL 2: iris ratio (eye-only gaze shifts) ───────────────────────
        if len(landmarks) >= 477:   # iris landmarks available
            l_iris = self.get_iris_position(landmarks, self.LEFT_IRIS,  frame_shape)
            r_iris = self.get_iris_position(landmarks, self.RIGHT_IRIS, frame_shape)
            if l_iris is not None and r_iris is not None:
                lp = self.get_eye_points(landmarks, self.LEFT_EYE,  frame_shape)
                rp = self.get_eye_points(landmarks, self.RIGHT_EYE, frame_shape)
                l_ratio = self.calculate_iris_ratio(lp, l_iris)
                r_ratio = self.calculate_iris_ratio(rp, r_iris)
                avg = (l_ratio + r_ratio) / 2.0
                if avg < self.IRIS_RATIO_MIN or avg > self.IRIS_RATIO_MAX:
                    return False   # eyes drifted off-centre

        return True

    def _smooth(self, raw: bool) -> bool:
        """Majority vote over last 3 frames to dampen single-frame noise."""
        self._gaze_window.append(raw)
        if len(self._gaze_window) < 2:
            return raw
        trues  = sum(1 for v in self._gaze_window if v)
        falses = len(self._gaze_window) - trues
        return trues >= falses

    def _make_result(self, landmarks, frame_shape):
        """Run signals, smooth, and return result dict."""
        looking_raw    = self._is_looking(landmarks, frame_shape)
        looking        = self._smooth(looking_raw)
        gaze_direction = "center" if looking else "away"
        self.gaze_history.append(gaze_direction)
        return {"gaze": gaze_direction, "looking_at_camera": looking}

    # =========================================================================
    # Public API used by inference.py (shared landmarks from DrowsinessDetector)
    # =========================================================================

    def analyze_gaze_from_landmarks(self, landmarks, frame_shape):
        """Called by single-pass pipeline with pre-detected landmarks."""
        if landmarks is None or len(landmarks) < 6:
            return None
        return self._make_result(landmarks, frame_shape)

    # =========================================================================
    # Public API — standalone per-frame (used in analyze_video below)
    # =========================================================================

    def process_frame(self, rgb_frame, frame_shape):
        lms = self._detect_classic(rgb_frame) or self._detect_tasks(rgb_frame)
        return self._make_result(lms, frame_shape) if lms else None

    def _detect_classic(self, rgb_frame):
        if not self.use_face_mesh or self.face_mesh is None:
            return None
        try:
            r = self.face_mesh.process(rgb_frame)
            if r.multi_face_landmarks:
                return r.multi_face_landmarks[0].landmark
        except Exception:
            pass
        return None

    def _detect_tasks(self, rgb_frame):
        if not self.use_tasks or self.face_landmarker is None or _mp_module is None:
            return None
        try:
            img = _mp_module.Image(image_format=_mp_module.ImageFormat.SRGB, data=rgb_frame)
            r   = self.face_landmarker.detect(img)
            if r.face_landmarks:
                return r.face_landmarks[0]
        except Exception:
            pass
        return None

    # =========================================================================
    # Standalone video analysis
    # =========================================================================

    def analyze_video(self, video_path):
        if not self.use_mediapipe:
            return {
                "error": "Eye tracking not available",
                "eye_contact_percentage": 0,
                "looking_away_count": 0,
                "gaze_stability": "N/A",
                "coaching_feedback": ["Eye tracking is currently unavailable."]
            }

        cap = cv2.VideoCapture(video_path)
        frames_with_face = center_frames = consecutive_away = frame_count = 0
        away_events = []
        skip = 3
        print(f"👁️ Eye tracking: {video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % skip != 0:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.process_frame(rgb, frame.shape)
            if res:
                frames_with_face += 1
                if res["looking_at_camera"]:
                    center_frames  += 1
                    consecutive_away = 0
                else:
                    consecutive_away += 1
                    if consecutive_away == 5:
                        away_events.append(frame_count)

        cap.release()
        if frames_with_face == 0:
            return {
                "error": "No faces detected",
                "eye_contact_percentage": 0,
                "looking_away_count": 0,
                "gaze_stability": "Unknown",
                "coaching_feedback": ["Could not detect face. Ensure good lighting."]
            }

        pct   = round(center_frames / frames_with_face * 100, 1)
        count = len(away_events)
        stab  = ("Excellent" if pct >= 70 else "Good" if pct >= 50
                 else "Fair" if pct >= 30 else "Needs Improvement")
        return {
            "eye_contact_percentage": pct,
            "looking_away_count":     count,
            "gaze_stability":         stab,
            "coaching_feedback":      self.generate_coaching_feedback(pct, count, stab)
        }

    def generate_coaching_feedback(self, pct, away_count, stability):
        fb = []
        if pct >= 70:
            fb.append(f"Excellent eye contact! You maintained {pct}% eye contact.")
        elif pct >= 50:
            fb.append(f"Good eye contact at {pct}%. Aim for 70%+ for stronger engagement.")
        elif pct >= 30:
            fb.append(f"Eye contact was {pct}%. Practice holding eye contact longer.")
        else:
            fb.append(f"Low eye contact ({pct}%). Focus on looking at the camera consistently.")

        if away_count == 0:
            fb.append("Great focus! You didn't look away during the response.")
        elif away_count <= 2:
            fb.append(f"You looked away {away_count} time(s). Try to minimise this.")
        else:
            fb.append(f"You looked away {away_count} times. Work on steady eye contact.")

        if pct < 60:
            fb.append("💡 Tip: Look directly at the camera lens as if making eye contact with the interviewer.")
        return fb

    def __del__(self):
        for attr in ('face_mesh', 'face_landmarker'):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
