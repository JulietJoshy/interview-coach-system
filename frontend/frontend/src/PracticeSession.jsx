import { useState, useRef, useEffect } from "react";
import { useReactMediaRecorder } from "react-media-recorder";
import axios from "axios";

const VideoPreview = ({ stream }) => {
    const videoRef = useRef(null);
    useEffect(() => {
        if (videoRef.current && stream) videoRef.current.srcObject = stream;
    }, [stream]);
    if (!stream) return null;
    return <video ref={videoRef} autoPlay muted />;
};

const PracticeSession = ({ onBack }) => {
    const [analysis, setAnalysis] = useState(null);
    const [emotionData, setEmotionData] = useState(null);
    const [eyeTrackingData, setEyeTrackingData] = useState(null);
    const [drowsinessData, setDrowsinessData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [isSpeaking, setIsSpeaking] = useState(false);
    const [resumeQuestions, setResumeQuestions] = useState([]);
    const [loadingResumeQ, setLoadingResumeQ] = useState(false);
    const [jobRole, setJobRole] = useState("");
    const [file, setFile] = useState(null);
    const [fileName, setFileName] = useState("No file chosen");
    const [setupDone, setSetupDone] = useState(false);
    const [questions, setQuestions] = useState([
        "Tell me about yourself.",
        "What is your biggest weakness?",
        "Why should we hire you?",
        "Describe a challenging technical problem you solved.",
        "Where do you see yourself in 5 years?"
    ]);
    const [selectedQuestion, setSelectedQuestion] = useState("Tell me about yourself.");

    const handleResumeUpload = async () => {
        if (!jobRole) { alert("⚠️ Please enter a Target Job Role"); return; }
        const formData = new FormData();
        formData.append("job_role", jobRole);
        if (file) formData.append("file", file);
        try {
            setLoading(true);
            await axios.post("http://127.0.0.1:8000/upload_resume", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });
            setLoading(false);
            setSetupDone(true);
            if (file) { alert("✅ Resume & Role Analyzed! Starting Interview..."); fetchResumeQuestions(); }
            else { alert(`✅ Role '${jobRole}' Set! Starting Interview...`); }
        } catch (error) {
            setLoading(false);
            alert(error.message ? `❌ Network Error: ${error.message}` : "❌ Setup failed.");
        }
    };

    const fetchResumeQuestions = async () => {
        setLoadingResumeQ(true);
        try {
            const res = await axios.post("http://127.0.0.1:8000/generate_resume_questions");
            if (res.data.status === "success" && res.data.questions.length > 0)
                setResumeQuestions(res.data.questions);
        } catch (e) { console.warn(e); }
        finally { setLoadingResumeQ(false); }
    };

    const handleOnStop = async (blobUrl, blob) => {
        setLoading(true);
        setAnalysis(null); setEmotionData(null); setEyeTrackingData(null); setDrowsinessData(null);
        window.speechSynthesis.cancel();
        try {
            const videoFile = new File([blob], "interview.webm", { type: "video/webm" });
            const formData = new FormData();
            formData.append("file", videoFile);
            formData.append("question", selectedQuestion);
            const response = await axios.post("http://127.0.0.1:8000/process-video", formData,
                { headers: { "Content-Type": "multipart/form-data" } });
            if (response.data.status === "error" || response.data.error) {
                alert(`❌ Backend Error: ${response.data.message || response.data.error}`); return;
            }
            setAnalysis(response.data.ai_analysis);
            if (response.data.emotion_analysis && !response.data.emotion_analysis.error) setEmotionData(response.data.emotion_analysis);
            if (response.data.eye_tracking && !response.data.eye_tracking.error) setEyeTrackingData(response.data.eye_tracking);
            if (response.data.drowsiness && !response.data.drowsiness.error) setDrowsinessData(response.data.drowsiness);
        } catch (err) { console.error(err); alert("Error processing video."); }
        finally { setLoading(false); }
    };

    // useReactMediaRecorder is now ONLY mounted when PracticeSession is rendered
    const { status, startRecording, stopRecording, mediaBlobUrl, previewStream } =
        useReactMediaRecorder({
            video: { width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: true,
            blobPropertyBag: { type: "video/webm" },
            videoBitsPerSecond: 600000,
            audioBitsPerSecond: 128000,
            onStop: handleOnStop
        });

    const speakText = (text) => {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.onend = () => setIsSpeaking(false);
        window.speechSynthesis.speak(utterance);
        setIsSpeaking(true);
    };

    useEffect(() => {
        if (setupDone && selectedQuestion) {
            setTimeout(() => window.speechSynthesis.speak(new SpeechSynthesisUtterance(selectedQuestion)), 500);
        }
    }, [selectedQuestion, setupDone]);

    // ── SETUP SCREEN ────────────────────────────────────────────────────────────
    if (!setupDone) {
        return (
            <div className="hero-section">
                <h1>Configure Your Interview</h1>
                <p className="hero-subtitle">
                    Practice with our AI-powered coach. Get real-time feedback on your tone, confidence, and answers.
                </p>
                <div className="session-card">
                    <div className="session-card-header">
                        <div className="header-icon">🎬</div>
                        <h2>Configure Session</h2>
                    </div>
                    <div className="session-card-body">
                        <div className="form-section">
                            <div className="form-group">
                                <label className="form-label">Target Job Role</label>
                                <input type="text" placeholder="e.g. Senior React Developer" value={jobRole}
                                    onChange={(e) => setJobRole(e.target.value)}
                                    onKeyDown={(e) => e.key === "Enter" && handleResumeUpload()} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Resume (Optional)</label>
                                <div className="file-upload-wrapper" onClick={() => document.getElementById('resume-input').click()}>
                                    <span className="file-label">Choose File</span>
                                    <span className="file-name">{fileName}</span>
                                    <input id="resume-input" type="file" accept=".pdf"
                                        onChange={(e) => { setFile(e.target.files[0]); setFileName(e.target.files[0]?.name || "No file chosen"); }} />
                                </div>
                            </div>
                        </div>
                        <div className="action-section">
                            <p>Ready to start? We'll generate questions based on your role.</p>
                            <button className="begin-btn" onClick={handleResumeUpload} disabled={loading}>
                                {loading ? "Setting up..." : <>Begin Interview &nbsp;›</>}
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    // ── INTERVIEW SCREEN ─────────────────────────────────────────────────────────
    return (
        <div className="interview-wrapper">
            <div className="interview-layout">
                <div className="left-panel">
                    <div className="current-question-display">
                        <small>Current Topic</small>
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

                    {analysis && (
                        <div className="results fade-in">
                            <div className="score-header">
                                <h2>Score: {analysis.rating}/100</h2>
                                <button onClick={() => speakText(`You scored ${analysis.rating}. ${analysis.feedback}`)} className="speak-btn">
                                    {isSpeaking ? "⏹️ Stop" : "🔊 Listen"}
                                </button>
                            </div>
                            <p><strong>💡 Feedback:</strong> {analysis.feedback}</p>
                            <div className="improvement-box">
                                <p><strong>✨ Better Way to Say It:</strong> "{analysis.improved_answer}"</p>
                            </div>
                            {analysis.follow_up_question && (
                                <button className="follow-up-btn" onClick={() => {
                                    setQuestions(prev => [...prev, analysis.follow_up_question]);
                                    setSelectedQuestion(analysis.follow_up_question);
                                    setAnalysis(null);
                                }}>
                                    ➡️ Try Follow-up: "{analysis.follow_up_question}"
                                </button>
                            )}
                        </div>
                    )}

                    {emotionData && (
                        <div className="results fade-in emotion-box">
                            <div className="score-header">
                                <h3>🎭 Facial Analysis</h3>
                                <span className="emotion-badge">{emotionData.dominant_emotion}</span>
                            </div>
                            <div className="emotion-feedback">
                                {(emotionData.coaching_feedback || []).map((tip, i) => <p key={i}>💡 {tip}</p>)}
                            </div>
                            <div className="emotion-breakdown">
                                <small>Emotion Breakdown:</small>
                                <div className="graph-container">
                                    {Object.entries(emotionData.emotion_breakdown || {}).map(([emotion, score]) =>
                                        score > 5 && <div key={emotion} className="graph-bar" style={{ width: `${score}%` }}>{emotion} ({score}%)</div>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}

                    {eyeTrackingData && !eyeTrackingData.error && (
                        <div className="results fade-in eye-tracking-box">
                            <div className="score-header">
                                <h3>👁️ Eye Contact Analysis</h3>
                                <span className="eye-badge">{eyeTrackingData.gaze_stability}</span>
                            </div>
                            <div className="eye-metrics">
                                <div className="metric-item"><span className="metric-label">Eye Contact</span><span className="metric-value">{eyeTrackingData.eye_contact_percentage}%</span></div>
                                <div className="metric-item"><span className="metric-label">Looking Away</span><span className="metric-value">{eyeTrackingData.looking_away_count}×</span></div>
                            </div>
                            <div className="eye-feedback">{(eyeTrackingData.coaching_feedback || []).map((tip, i) => <p key={i}>💡 {tip}</p>)}</div>
                            <div className="eye-contact-bar">
                                <small>Eye Contact Progress:</small>
                                <div className="progress-bar-container">
                                    <div className="progress-bar-fill" style={{ width: `${eyeTrackingData.eye_contact_percentage}%`, backgroundColor: eyeTrackingData.eye_contact_percentage >= 70 ? '#16a34a' : eyeTrackingData.eye_contact_percentage >= 50 ? '#ca8a04' : '#dc2626' }}>
                                        {eyeTrackingData.eye_contact_percentage}%
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {drowsinessData && !drowsinessData.error && (
                        <div className="results fade-in drowsiness-box">
                            <div className="score-header">
                                <h3>😴 Alertness Analysis</h3>
                                <span className={`drowsiness-badge ${drowsinessData.drowsiness_level === 'Alert' ? 'badge-alert' : drowsinessData.drowsiness_level === 'Mild Drowsiness' ? 'badge-mild' : drowsinessData.drowsiness_level === 'Moderate Drowsiness' ? 'badge-moderate' : 'badge-severe'}`}>
                                    {drowsinessData.drowsiness_level}
                                </span>
                            </div>
                            <div className="drowsiness-metrics">
                                <div className="metric-item"><span className="metric-label">Alertness</span><span className="metric-value">{drowsinessData.alertness_score}/100</span></div>
                                <div className="metric-item"><span className="metric-label">Yawns</span><span className="metric-value">{drowsinessData.yawn_count}</span></div>
                                <div className="metric-item"><span className="metric-label">Blink Rate</span><span className="metric-value">{drowsinessData.blink_rate}/min</span></div>
                                <div className="metric-item"><span className="metric-label">Eyes Closed</span><span className="metric-value">{drowsinessData.perclos}%</span></div>
                            </div>
                            <div className="drowsiness-feedback">{(drowsinessData.coaching_feedback || []).map((tip, i) => <p key={i}>{tip}</p>)}</div>
                        </div>
                    )}
                </div>

                <div className="right-panel">
                    <h3>📌 Interview Questions</h3>
                    <div className="questions-list">
                        {questions.map((q, i) => (
                            <div key={i} className={`question-item ${selectedQuestion === q ? 'active' : ''}`}
                                onClick={() => { setSelectedQuestion(q); setAnalysis(null); window.speechSynthesis.cancel(); }}>
                                <h4>{i + 1}. {q}</h4>
                            </div>
                        ))}
                    </div>
                    {loadingResumeQ && <div className="resume-questions-section"><h3>📄 Resume Questions</h3><p className="loading-resume-q">Generating...</p></div>}
                    {resumeQuestions.length > 0 && (
                        <div className="resume-questions-section">
                            <h3>📄 Resume-Based Questions</h3>
                            <p className="resume-q-hint">Generated from your resume & target role</p>
                            <div className="questions-list">
                                {resumeQuestions.map((q, i) => (
                                    <div key={`resume-${i}`} className={`question-item resume-q ${selectedQuestion === q ? 'active' : ''}`}
                                        onClick={() => { setSelectedQuestion(q); setAnalysis(null); window.speechSynthesis.cancel(); }}>
                                        <h4>{i + 1}. {q}</h4>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default PracticeSession;
