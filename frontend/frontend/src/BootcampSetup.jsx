import { useState } from "react";
import axios from "axios";

const BootcampSetup = ({ onPlanGenerated, onBack }) => {
    const [jobRole, setJobRole] = useState("");
    const [days, setDays] = useState(5);
    const [file, setFile] = useState(null);
    const [fileName, setFileName] = useState("No file chosen");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    const dayDescriptions = {
        1: "Quick crash course – HR & behavioral only",
        2: "HR basics + Core technical concepts",
        3: "HR + Technical + Problem solving",
        4: "Full prep minus the final mock",
        5: "Complete 5-day bootcamp (recommended)",
    };

    const handleGenerate = async () => {
        if (!jobRole.trim()) {
            setError("⚠️ Please enter a Target Job Role.");
            return;
        }
        setError("");
        setLoading(true);

        try {
            let resumeText = "";
            let generatedResumeQs = [];

            if (file) {
                resumeText = `Resume uploaded: ${file.name}`;

                // 1. Upload resume to set context for questions
                try {
                    const ctxFormData = new FormData();
                    ctxFormData.append("job_role", jobRole);
                    ctxFormData.append("file", file);
                    await axios.post("http://127.0.0.1:8000/upload_resume", ctxFormData, {
                        headers: { "Content-Type": "multipart/form-data" },
                    });

                    // 2. Generate resume questions
                    const qRes = await axios.post("http://127.0.0.1:8000/generate_resume_questions");
                    if (qRes.data.status === "success" && qRes.data.questions?.length > 0) {
                        generatedResumeQs = qRes.data.questions;
                    }
                } catch (err) {
                    console.warn("Could not generate resume questions:", err);
                }
            }

            // 3. Generate the bootcamp plan
            const formData = new FormData();
            formData.append("job_role", jobRole);
            formData.append("days", days);
            formData.append("resume_text", resumeText);

            const response = await axios.post(
                "http://127.0.0.1:8000/generate_bootcamp_plan",
                formData,
                { headers: { "Content-Type": "multipart/form-data" } }
            );

            if (response.data.status === "success") {
                onPlanGenerated(response.data.plan, jobRole, generatedResumeQs);
            } else {
                setError(`❌ ${response.data.message || "Failed to generate plan."}`);
            }
        } catch (err) {
            console.error(err);
            setError("❌ Network error. Make sure the backend is running.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bootcamp-setup-page">
            <div className="hero-section" style={{ paddingBottom: "20px" }}>
                <h1>🚀 AI Interview Bootcamp</h1>
                <p className="hero-subtitle">
                    Tell us your target role and interview date. We'll generate a
                    personalized day-by-day preparation roadmap.
                </p>
            </div>

            <div className="session-card bootcamp-card">
                <div className="session-card-header">
                    <div className="header-icon">🗓️</div>
                    <h2>Configure Your Bootcamp</h2>
                </div>

                <div className="bootcamp-form">
                    {/* Job Role */}
                    <div className="form-group">
                        <label className="form-label">Target Job Role</label>
                        <input
                            type="text"
                            placeholder="e.g. Python Developer, Data Scientist, Frontend Engineer"
                            value={jobRole}
                            onChange={(e) => setJobRole(e.target.value)}
                            onKeyDown={(e) => e.key === "Enter" && handleGenerate()}
                        />
                    </div>

                    {/* Days Slider */}
                    <div className="form-group">
                        <label className="form-label">Days Until Interview</label>
                        <div className="days-slider-wrapper">
                            <div className="days-display">
                                <span className="days-number">{days}</span>
                                <span className="days-label">
                                    {days === 1 ? "day" : "days"}
                                </span>
                            </div>
                            <input
                                type="range"
                                min="1"
                                max="5"
                                value={days}
                                onChange={(e) => setDays(parseInt(e.target.value))}
                                className="days-slider"
                            />
                            <div className="slider-ticks">
                                {[1, 2, 3, 4, 5].map((d) => (
                                    <span
                                        key={d}
                                        className={`tick ${d <= days ? "tick-active" : ""}`}
                                        onClick={() => setDays(d)}
                                    >
                                        {d}
                                    </span>
                                ))}
                            </div>
                        </div>
                        <p className="days-description">{dayDescriptions[days]}</p>
                    </div>

                    {/* Resume (Optional) */}
                    <div className="form-group">
                        <label className="form-label">Resume (Optional)</label>
                        <div
                            className="file-upload-wrapper"
                            onClick={() =>
                                document.getElementById("bootcamp-resume-input").click()
                            }
                        >
                            <span className="file-label">Choose File</span>
                            <span className="file-name">{fileName}</span>
                            <input
                                id="bootcamp-resume-input"
                                type="file"
                                accept=".pdf"
                                onChange={(e) => {
                                    setFile(e.target.files[0]);
                                    setFileName(e.target.files[0]?.name || "No file chosen");
                                }}
                            />
                        </div>
                        <p style={{ fontSize: "0.78rem", color: "#555", margin: "4px 0 0" }}>
                            Upload your resume for more tailored questions
                        </p>
                    </div>

                    {error && <p className="bootcamp-error">{error}</p>}

                    {/* Day Preview */}
                    <div className="day-preview-strip">
                        {Array.from({ length: days }, (_, i) => i + 1).map((d) => {
                            const titles = [
                                "HR & Behavioral",
                                "Core Technical",
                                "Problem Solving",
                                "Advanced Tech",
                                "Full Mock",
                            ];
                            const icons = ["💬", "🧠", "🛠️", "⚙️", "🎯"];
                            return (
                                <div key={d} className="day-preview-item">
                                    <span className="day-preview-icon">{icons[d - 1]}</span>
                                    <span className="day-preview-label">Day {d}</span>
                                    <span className="day-preview-title">{titles[d - 1]}</span>
                                </div>
                            );
                        })}
                    </div>

                    <div className="bootcamp-actions">
                        <button className="back-btn" onClick={onBack}>
                            ← Back
                        </button>
                        <button
                            className="begin-btn bootcamp-generate-btn"
                            onClick={handleGenerate}
                            disabled={loading}
                        >
                            {loading ? (
                                <>
                                    <span className="spinner" /> Generating Roadmap...
                                </>
                            ) : (
                                <>Generate My {days}-Day Roadmap &nbsp;›</>
                            )}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default BootcampSetup;
