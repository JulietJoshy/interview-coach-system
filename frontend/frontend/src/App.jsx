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
  const [loading, setLoading] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  
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
        if(file) alert("✅ Resume & Role Analyzed! Starting Interview...");
        else alert(`✅ Role '${jobRole}' Set! Starting Interview...`);

    } catch (error) {
        console.error(error);
        setLoading(false);
        alert("❌ Setup failed. Check Backend Console.");
    }
  };

  // --- 2. VIDEO SUBMIT ---
  const handleOnStop = async (blobUrl, blob) => {
    setLoading(true);
    setAnalysis(null);
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

      setAnalysis(response.data.ai_analysis);

    } catch (error) {
      console.error("❌ Upload Error:", error);
      alert("Error processing video. Check backend logs.");
    } finally {
      setLoading(false);
    }
  };

  const { status, startRecording, stopRecording, mediaBlobUrl, previewStream } =
    useReactMediaRecorder({
      video: true, audio: true, blobPropertyBag: { type: "video/webm" }, onStop: handleOnStop
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

      {/* --- SCREEN 1: SETUP --- */ }
      {!resumeUploaded ? (
        <div className="setup-box">
            <h2>📝 Step 1: Setup</h2>
            <p className="subtitle">Enter a role to start. Upload a resume for better results (Optional).</p>
            
            <input 
                type="text" 
                placeholder="Target Job Role (e.g. Data Scientist)" 
                value={jobRole} 
                onChange={(e) => setJobRole(e.target.value)}
            />
            
            <div className="file-input-wrapper">
                <label>Upload Resume (Optional):</label>
                <input 
                    type="file" accept=".pdf"
                    onChange={(e) => setFile(e.target.files[0])}
                />
            </div>

            <button onClick={handleResumeUpload} className="start-btn full-width">
                {loading ? "Setting up..." : "Start Interview 🚀"}
            </button>
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
            </div>

        </div>
      )}
    </div>
  );
};

export default App;