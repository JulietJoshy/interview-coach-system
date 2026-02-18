import { useState, useRef, useEffect } from "react";
import { useReactMediaRecorder } from "react-media-recorder";
import axios from "axios";
import "./App.css";

const VideoPreview = ({ stream }) => {
  const videoRef = useRef(null);
  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);
  if (!stream) return null;
  return <video ref={videoRef} autoPlay muted />;
};

const App = () => {
  const [analysis, setAnalysis] = useState(null);
  const [emotionData, setEmotionData] = useState(null);
  const [eyeTrackingData, setEyeTrackingData] = useState(null);
  const [drowsinessData, setDrowsinessData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [resumeQuestions, setResumeQuestions] = useState([]);
  const [loadingResumeQ, setLoadingResumeQ] = useState(false);

  // Setup State
  const [jobRole, setJobRole] = useState("");
  const [file, setFile] = useState(null);
  const [resumeUploaded, setResumeUploaded] = useState(false);

  // Questions State
  const [questions, setQuestions] = useState([
    "Tell me about yourself.",
    "What is your biggest weakness?",
    "Why should we hire you?",
    "Describe a challenging technical problem you solved.",
    "Where do you see yourself in 5 years?"
  ]);
  const [selectedQuestion, setSelectedQuestion] = useState(questions[0]);

  // --- 1. SETUP (RESUME OPTIONAL) ---
  const handleResumeUpload = async () => {
    if (!jobRole) {
      alert("⚠️ Please enter a Target Job Role (e.g. 'Data Scientist')");
      return;
    }

    const formData = new FormData();
    formData.append("job_role", jobRole);

    // Only append file if one was selected
    if (file) {
      formData.append("file", file);
    }

    try {
      setLoading(true);
      await axios.post("http://127.0.0.1:8000/upload_resume", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setLoading(false);
      setResumeUploaded(true);

      // Dynamic Alert
      if (file) {
        alert("✅ Resume & Role Analyzed! Starting Interview...");
        // Fetch resume-based questions in background
        fetchResumeQuestions();
      } else {
        alert(`✅ Role '${jobRole}' Set! Starting Interview...`);
      }

    } catch (error) {
      console.error(error);
      setLoading(false);

      let message = "❌ Setup failed. Check Backend Console.";
      if (error.response && error.response.data && error.response.data.message) {
        message = `❌ Server Error: ${error.response.data.message}`;
      } else if (error.message) {
        message = `❌ Network Error: ${error.message}`;
      }
      alert(message);
    }
  };

  // --- FETCH RESUME QUESTIONS ---
  const fetchResumeQuestions = async () => {
    setLoadingResumeQ(true);
    try {
      const response = await axios.post("http://127.0.0.1:8000/generate_resume_questions");
      if (response.data.status === "success" && response.data.questions.length > 0) {
        setResumeQuestions(response.data.questions);
        console.log("✅ Resume questions received:", response.data.questions);
      }
    } catch (error) {
      console.warn("⚠️ Could not fetch resume questions:", error);
    } finally {
      setLoadingResumeQ(false);
    }
  };

  // --- 2. VIDEO SUBMIT ---
  const handleOnStop = async (blobUrl, blob) => {
    setLoading(true);
    setAnalysis(null);
    setEmotionData(null);
    setEyeTrackingData(null);
    setDrowsinessData(null);
    window.speechSynthesis.cancel();
    setIsSpeaking(false);

    try {
      const file = new File([blob], "interview.webm", { type: "video/webm" });
      const formData = new FormData();
      formData.append("file", file);
      formData.append("question", selectedQuestion);

      const response = await axios.post(
        "http://127.0.0.1:8000/process-video",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      console.log("✅ Server Response:", response.data);

      if (response.data.status === "error" || response.data.error) {
        alert(`❌ Backend Error: ${response.data.message || response.data.error}`);
        return;
      }

      setAnalysis(response.data.ai_analysis);

      // Handle Emotion Data (safely)
      if (response.data.emotion_analysis && !response.data.emotion_analysis.error) {
        setEmotionData(response.data.emotion_analysis);
      } else {
        console.warn("⚠️ Emotion analysis failed or missing:", response.data.emotion_analysis);
        setEmotionData(null);
        // Optional: Alert the user about partial success
        if (response.data.emotion_analysis?.error) {
          alert(`⚠️ Gemini Analysis Success, but Facial Analysis Failed: ${response.data.emotion_analysis.error}`);
        }
      }

      // Handle Eye Tracking Data
      if (response.data.eye_tracking && !response.data.eye_tracking.error) {
        setEyeTrackingData(response.data.eye_tracking);
      } else {
        console.warn("⚠️ Eye tracking failed or missing:", response.data.eye_tracking);
        setEyeTrackingData(null);
      }

      // Handle Drowsiness Data
      if (response.data.drowsiness && !response.data.drowsiness.error) {
        setDrowsinessData(response.data.drowsiness);
      } else {
        console.warn("⚠️ Drowsiness analysis failed or missing:", response.data.drowsiness);
        setDrowsinessData(null);
      }

    } catch (error) {
      console.error("❌ Upload Error:", error);
      alert("Error processing video. Check backend logs.");
    } finally {
      setLoading(false);
    }
  };

  const { status, startRecording, stopRecording, mediaBlobUrl, previewStream } =
    useReactMediaRecorder({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: true,
      blobPropertyBag: { type: "video/webm" },
      videoBitsPerSecond: 600000, // 600 kbps (reduced from default ~2500 kbps)
      audioBitsPerSecond: 128000, // 128 kbps
      onStop: handleOnStop
    });

  const speakText = (text) => {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.onend = () => setIsSpeaking(false);
    window.speechSynthesis.speak(utterance);
    setIsSpeaking(true);
  };

  // Auto-speak question when changed
  useEffect(() => {
    if (resumeUploaded && selectedQuestion) {
      setTimeout(() => {
        const utterance = new SpeechSynthesisUtterance(selectedQuestion);
        window.speechSynthesis.speak(utterance);
      }, 500);
    }
  }, [selectedQuestion, resumeUploaded]);

  return (
    <div className="container">
      <h1>🤖 AI Interview Coach</h1>

      {/* --- SCREEN 1: SETUP --- */}
      {!resumeUploaded ? (
        <div className="setup-layout">
          {/* LEFT: Setup Form */}
          <div className="setup-form-panel">
            <div className="setup-form-header">
              <h2>Configure Your Interview</h2>
              <p className="subtitle">Personalize your practice session for targeted feedback and coaching.</p>
            </div>

            <div className="setup-form-body">
              <div className="form-group">
                <label className="form-label">🎯 Target Role</label>
                <input
                  type="text"
                  placeholder="e.g. Software Engineer, Data Scientist, Product Manager"
                  value={jobRole}
                  onChange={(e) => setJobRole(e.target.value)}
                />
              </div>

              <div className="form-group">
                <label className="form-label">📄 Resume (Optional)</label>
                <p className="form-hint">Upload for role-specific feedback and tailored questions.</p>
                <input
                  type="file" accept=".pdf"
                  onChange={(e) => setFile(e.target.files[0])}
                />
              </div>

              <button onClick={handleResumeUpload} className="start-btn full-width">
                {loading ? "Setting up..." : "Begin Practice Session →"}
              </button>
            </div>
          </div>

          {/* RIGHT: Feature Highlights */}
          <div className="setup-features-panel">
            <h3 className="features-title">What You'll Get</h3>
            <div className="feature-list">
              <div className="feature-card">
                <div className="feature-icon">🧠</div>
                <div className="feature-info">
                  <h4>AI Content Analysis</h4>
                  <p>Get scored feedback on your answer quality, structure, and relevance.</p>
                </div>
              </div>
              <div className="feature-card">
                <div className="feature-icon">🎭</div>
                <div className="feature-info">
                  <h4>Emotion Recognition</h4>
                  <p>Real-time facial expression analysis to help you appear confident.</p>
                </div>
              </div>
              <div className="feature-card">
                <div className="feature-icon">👁️</div>
                <div className="feature-info">
                  <h4>Eye Contact Tracking</h4>
                  <p>Measure how well you maintain eye contact during your answer.</p>
                </div>
              </div>
              <div className="feature-card">
                <div className="feature-icon">😴</div>
                <div className="feature-info">
                  <h4>Alertness Detection</h4>
                  <p>Detect drowsiness, yawning, and fatigue to ensure peak performance.</p>
                </div>
              </div>
            </div>
            <div className="features-footer">
              <p>💡 Powered by CNN emotion models, computer vision, and Google Gemini AI</p>
            </div>
          </div>
        </div>
      ) : (
        /* --- SCREEN 2: INTERVIEW (SPLIT VIEW) --- */
        <div className="interview-layout">

          {/* LEFT: Video & Feedback */}
          <div className="left-panel">
            <div className="current-question-display">
              <small>CURRENT TOPIC:</small>
              <h2>"{selectedQuestion}"</h2>
            </div>

            <div className="video-box">
              {status === "idle" && <div className="placeholder-text">Ready? Press Start below.</div>}
              {status === "recording" && (
                <div className="live-preview">
                  <div className="rec-dot">🔴 REC</div>
                  <VideoPreview stream={previewStream} />
                </div>
              )}
              {status === "stopped" && mediaBlobUrl && !loading && (
                <video src={mediaBlobUrl} controls autoPlay />
              )}
            </div>

            <div className="controls">
              {status !== "recording" && !loading && (
                <button onClick={startRecording} className="start-btn">🔴 Start Answer</button>
              )}
              {status === "recording" && (
                <button onClick={stopRecording} className="stop-btn">⏹ Stop & Submit</button>
              )}
            </div>

            {loading && <div className="loading-text">🧠 AI is analyzing your confidence, tone, and content...</div>}

            {/* RESULTS */}
            {analysis && (
              <div className="results fade-in">
                <div className="score-header">
                  <h2>Score: {analysis.rating}/100</h2>
                  <button
                    onClick={() => speakText(`You scored ${analysis.rating}. ${analysis.feedback}`)}
                    className="speak-btn"
                  >
                    {isSpeaking ? "⏹️ Stop" : "🔊 Listen"}
                  </button>
                </div>

                <p><strong>💡 Feedback:</strong> {analysis.feedback}</p>

                <div className="improvement-box">
                  <p><strong>✨ Better Way to Say It:</strong> "{analysis.improved_answer}"</p>
                </div>

                {analysis.follow_up_question && (
                  <button
                    className="follow-up-btn"
                    onClick={() => {
                      setQuestions(prev => [...prev, analysis.follow_up_question]);
                      setSelectedQuestion(analysis.follow_up_question);
                      setAnalysis(null);
                    }}
                  >
                    ➡️ Try Follow-up: "{analysis.follow_up_question}"
                  </button>
                )}
              </div>
            )}

            {/* EMOTION RESULTS */}
            {emotionData && (
              <div className="results fade-in emotion-box">
                <div className="score-header">
                  <h3>🎭 Facial Analysis</h3>
                  <span className="emotion-badge">{emotionData.dominant_emotion}</span>
                </div>

                <div className="emotion-feedback">
                  {(emotionData.coaching_feedback || []).map((tip, i) => (
                    <p key={i}>💡 {tip}</p>
                  ))}
                </div>

                <div className="emotion-breakdown">
                  <small>Emotion Breakdown:</small>
                  <div className="graph-container">
                    {Object.entries(emotionData.emotion_breakdown).map(([emotion, score]) => (
                      score > 5 && (
                        <div key={emotion} className="graph-bar" style={{ width: `${score}%` }}>
                          {emotion} ({score}%)
                        </div>
                      )
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* EYE TRACKING RESULTS */}
            {eyeTrackingData && !eyeTrackingData.error && (
              <div className="results fade-in eye-tracking-box">
                <div className="score-header">
                  <h3>👁️ Eye Contact Analysis</h3>
                  <span className="eye-badge">{eyeTrackingData.gaze_stability}</span>
                </div>

                <div className="eye-metrics">
                  <div className="metric-item">
                    <span className="metric-label">Eye Contact:</span>
                    <span className="metric-value">{eyeTrackingData.eye_contact_percentage}%</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">Looking Away:</span>
                    <span className="metric-value">{eyeTrackingData.looking_away_count} times</span>
                  </div>
                </div>

                <div className="eye-feedback">
                  {eyeTrackingData.coaching_feedback.map((tip, i) => (
                    <p key={i}>💡 {tip}</p>
                  ))}
                </div>

                {/* Visual progress bar for eye contact */}
                <div className="eye-contact-bar">
                  <small>Eye Contact Progress:</small>
                  <div className="progress-bar-container">
                    <div
                      className="progress-bar-fill"
                      style={{
                        width: `${eyeTrackingData.eye_contact_percentage}%`,
                        backgroundColor: eyeTrackingData.eye_contact_percentage >= 70 ? '#4CAF50' :
                          eyeTrackingData.eye_contact_percentage >= 50 ? '#FFC107' : '#FF5252'
                      }}
                    >
                      {eyeTrackingData.eye_contact_percentage}%
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* DROWSINESS RESULTS */}
            {drowsinessData && !drowsinessData.error && (
              <div className="results fade-in drowsiness-box">
                <div className="score-header">
                  <h3>😴 Alertness Analysis</h3>
                  <span className={`drowsiness-badge ${drowsinessData.drowsiness_level === 'Alert' ? 'badge-alert' :
                    drowsinessData.drowsiness_level === 'Mild Drowsiness' ? 'badge-mild' :
                      drowsinessData.drowsiness_level === 'Moderate Drowsiness' ? 'badge-moderate' : 'badge-severe'
                    }`}>{drowsinessData.drowsiness_level}</span>
                </div>

                <div className="drowsiness-metrics">
                  <div className="metric-item">
                    <span className="metric-label">Alertness Score:</span>
                    <span className="metric-value drowsiness-value">{drowsinessData.alertness_score}/100</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">Yawns:</span>
                    <span className="metric-value drowsiness-value">{drowsinessData.yawn_count}</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">Blink Rate:</span>
                    <span className="metric-value drowsiness-value">{drowsinessData.blink_rate}/min</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">Eyes Closed:</span>
                    <span className="metric-value drowsiness-value">{drowsinessData.perclos}%</span>
                  </div>
                </div>

                <div className="drowsiness-feedback">
                  {(drowsinessData.coaching_feedback || []).map((tip, i) => (
                    <p key={i}>{tip}</p>
                  ))}
                </div>

                {/* Alertness progress bar */}
                <div className="eye-contact-bar">
                  <small>Alertness Level:</small>
                  <div className="progress-bar-container">
                    <div
                      className="progress-bar-fill"
                      style={{
                        width: `${drowsinessData.alertness_score}%`,
                        backgroundColor: drowsinessData.alertness_score >= 80 ? '#4CAF50' :
                          drowsinessData.alertness_score >= 60 ? '#FFC107' :
                            drowsinessData.alertness_score >= 40 ? '#FF9800' : '#FF5252'
                      }}
                    >
                      {drowsinessData.alertness_score}%
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* RIGHT: Questions List */}
          <div className="right-panel">
            <h3>📌 Interview Questions</h3>
            <div className="questions-list">
              {questions.map((q, i) => (
                <div
                  key={i}
                  className={`question-item ${selectedQuestion === q ? 'active' : ''}`}
                  onClick={() => {
                    setSelectedQuestion(q);
                    setAnalysis(null);
                    window.speechSynthesis.cancel();
                  }}
                >
                  <h4>{i + 1}. {q}</h4>
                </div>
              ))}
            </div>

            {/* RESUME-BASED QUESTIONS */}
            {loadingResumeQ && (
              <div className="resume-questions-section">
                <h3>📄 Resume Questions</h3>
                <p className="loading-resume-q">Generating questions from your resume...</p>
              </div>
            )}

            {resumeQuestions.length > 0 && (
              <div className="resume-questions-section">
                <h3>📄 Resume-Based Questions</h3>
                <p className="resume-q-hint">Generated from your resume & target role</p>
                <div className="questions-list">
                  {resumeQuestions.map((q, i) => (
                    <div
                      key={`resume-${i}`}
                      className={`question-item resume-q ${selectedQuestion === q ? 'active' : ''}`}
                      onClick={() => {
                        setSelectedQuestion(q);
                        setAnalysis(null);
                        window.speechSynthesis.cancel();
                      }}
                    >
                      <h4>{i + 1}. {q}</h4>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

        </div>
      )}
    </div>
  );
};

export default App;