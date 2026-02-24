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

const BootcampDay = ({ dayItem, jobRole, onDayComplete, onBack }) => {
    const [currentQIndex, setCurrentQIndex] = useState(0);
    const [analysis, setAnalysis] = useState(null);
    const [loading, setLoading] = useState(false);
    const [isSpeaking, setIsSpeaking] = useState(false);
    const [answeredCount, setAnsweredCount] = useState(0);

    const questions = dayItem.questions;
    const currentQuestion = questions[currentQIndex];

    const handleOnStop = async (blobUrl, blob) => {
        setLoading(true);
        setAnalysis(null);
        window.speechSynthesis.cancel();

        try {
            const videoFile = new File([blob], `bootcamp_day${dayItem.day}.webm`, {
                type: "video/webm",
            });
            const formData = new FormData();
            formData.append("file", videoFile);
            formData.append("question", currentQuestion);
            formData.append("day", dayItem.day);
            formData.append("job_role", jobRole);

            const response = await axios.post(
                "http://127.0.0.1:8000/process_bootcamp_answer",
                formData,
                { headers: { "Content-Type": "multipart/form-data" } }
            );

            if (response.data.status === "error") {
                alert(`❌ ${response.data.message}`);
                return;
            }

            setAnalysis(response.data.ai_analysis);
            setAnsweredCount((prev) => prev + 1);
        } catch (err) {
            console.error(err);
            alert("Error processing video. Check backend logs.");
        } finally {
            setLoading(false);
        }
    };

    const { status, startRecording, stopRecording, mediaBlobUrl, previewStream } =
        useReactMediaRecorder({
            video: { width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: true,
            blobPropertyBag: { type: "video/webm" },
            onStop: handleOnStop,
        });

    const speakText = (text) => {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.onend = () => setIsSpeaking(false);
        window.speechSynthesis.speak(utterance);
        setIsSpeaking(true);
    };

    const goNextQuestion = () => {
        setAnalysis(null);
        setCurrentQIndex((prev) => prev + 1);
        window.speechSynthesis.cancel();
    };

    const dayIcons = ["💬", "🧠", "🛠️", "⚙️", "🎯"];

    return (
        <div className="bootcamp-day-page">
            {/* Day Header */}
            <div className="day-page-header">
                <button className="back-btn" onClick={onBack}>
                    ← Roadmap
                </button>
                <div className="day-page-title">
                    <span className="day-big-icon">{dayIcons[dayItem.day - 1]}</span>
                    <div>
                        <h2>Day {dayItem.day}: {dayItem.title}</h2>
                        <p>{dayItem.focus}</p>
                    </div>
                </div>
                {/* Question progress */}
                <div className="question-progress">
                    {questions.map((_, i) => (
                        <div
                            key={i}
                            className={`q-dot ${i < currentQIndex ? "q-dot-done" : i === currentQIndex ? "q-dot-current" : "q-dot-pending"}`}
                        />
                    ))}
                    <span className="q-progress-text">
                        {currentQIndex + 1}/{questions.length}
                    </span>
                </div>
            </div>

            <div className="day-practice-layout">
                {/* LEFT: Video + controls */}
                <div className="left-panel">
                    <div className="current-question-display">
                        <small>Question {currentQIndex + 1} of {questions.length}</small>
                        <h2>"{currentQuestion}"</h2>
                    </div>

                    <div className="video-box">
                        {status === "idle" && (
                            <div className="placeholder-text">Ready? Press Start below.</div>
                        )}
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
                            <button onClick={startRecording} className="start-btn">
                                🔴 Start Answer
                            </button>
                        )}
                        {status === "recording" && (
                            <button onClick={stopRecording} className="stop-btn">
                                ⏹ Stop & Submit
                            </button>
                        )}
                    </div>

                    {loading && (
                        <div className="loading-text">
                            🧠 AI is analyzing your Day {dayItem.day} answer...
                        </div>
                    )}

                    {/* AI FEEDBACK */}
                    {analysis && (
                        <div className="results fade-in">
                            <div className="score-header">
                                <h2>Score: {analysis.rating}/100</h2>
                                <button
                                    className="speak-btn"
                                    onClick={() =>
                                        speakText(`You scored ${analysis.rating}. ${analysis.feedback}`)
                                    }
                                >
                                    {isSpeaking ? "⏹️ Stop" : "🔊 Listen"}
                                </button>
                            </div>
                            <p>
                                <strong>💡 Feedback:</strong> {analysis.feedback}
                            </p>
                            <div className="improvement-box">
                                <p>
                                    <strong>✨ Better Answer:</strong> "{analysis.improved_answer}"
                                </p>
                            </div>

                            {/* Navigation */}
                            <div className="day-nav-actions">
                                {currentQIndex < questions.length - 1 ? (
                                    <button className="begin-btn" onClick={goNextQuestion}>
                                        Next Question →
                                    </button>
                                ) : (
                                    <button
                                        className="begin-btn complete-day-btn"
                                        onClick={onDayComplete}
                                    >
                                        ✅ Mark Day {dayItem.day} Complete
                                    </button>
                                )}
                            </div>
                        </div>
                    )}
                </div>

                {/* RIGHT: Questions sidebar */}
                <div className="right-panel">
                    <h3>📋 Day {dayItem.day} Questions</h3>
                    <div className="questions-list">
                        {questions.map((q, i) => (
                            <div
                                key={i}
                                className={`question-item ${i === currentQIndex ? "active" : ""} ${i < currentQIndex ? "question-answered" : ""}`}
                            >
                                <h4>
                                    {i < currentQIndex ? "✅" : i === currentQIndex ? "▶" : `${i + 1}.`}{" "}
                                    {q}
                                </h4>
                            </div>
                        ))}
                    </div>

                    <div className="day-tip-box">
                        <h4>💡 Day {dayItem.day} Tip</h4>
                        <p>
                            {dayItem.day === 1 &&
                                "Use the STAR method: Situation, Task, Action, Result for behavioral questions."}
                            {dayItem.day === 2 &&
                                "Explain concepts clearly — interviewers test depth of understanding, not memorization."}
                            {dayItem.day === 3 &&
                                "Think out loud. Interviewers value your reasoning process, not just the answer."}
                            {dayItem.day === 4 &&
                                "Relate answers to real projects you've worked on. Specifics impress more than theory."}
                            {dayItem.day === 5 &&
                                "Treat this like the real thing. Take a breath, stay calm, and be yourself."}
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default BootcampDay;
