import { useState } from "react";
import "./App.css";
import PracticeSession from "./PracticeSession";
import BootcampSetup from "./BootcampSetup";
import BootcampRoadmap from "./BootcampRoadmap";
import BootcampDay from "./BootcampDay";

const App = () => {
  // mode: 'home' | 'practice' | 'bootcamp-setup' | 'bootcamp-roadmap' | 'bootcamp-day'
  const [mode, setMode] = useState("home");

  // Bootcamp state
  const [bootcampPlan, setBootcampPlan] = useState(null);
  const [bootcampRole, setBootcampRole] = useState("");
  const [completedDays, setCompletedDays] = useState([]);
  const [activeDay, setActiveDay] = useState(null);

  const handlePlanGenerated = (plan, role) => {
    setBootcampPlan(plan);
    setBootcampRole(role);
    setCompletedDays([]);
    setMode("bootcamp-roadmap");
  };

  const handleStartDay = (dayItem) => {
    setActiveDay(dayItem);
    setMode("bootcamp-day");
  };

  const handleDayComplete = () => {
    setCompletedDays(prev => [...new Set([...prev, activeDay.day])]);
    setActiveDay(null);
    setMode("bootcamp-roadmap");
  };

  const goHome = () => {
    setMode("home");
    setBootcampPlan(null);
    setCompletedDays([]);
    setActiveDay(null);
  };

  return (
    <>
      {/* NAVBAR */}
      <nav className="navbar">
        <div className="navbar-brand">
          <div className="brand-icon">🎯</div>
          MockMate AI
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          {mode !== "home" && (
            <button
              className="back-btn"
              style={{ padding: "6px 14px", fontSize: "0.82rem" }}
              onClick={goHome}
            >
              ⌂ Home
            </button>
          )}
          <span className="navbar-badge">Interview Coach v1.0</span>
        </div>
      </nav>

      <div className="container">

        {/* HOME */}
        {mode === "home" && (
          <div className="hero-section">
            <h1>Master Your Interview</h1>
            <p className="hero-subtitle">
              Practice with our AI-powered coach. Get real-time feedback on your
              tone, confidence, and answers.
            </p>
            <div className="home-mode-cards">
              <div className="mode-card" onClick={() => setMode("practice")}>
                <div className="mode-card-icon">🎙️</div>
                <h3>Single Practice Session</h3>
                <p>Jump straight into a practice interview. Get AI feedback on any question instantly.</p>
                <span className="mode-card-btn">Start Session →</span>
              </div>
              <div className="mode-card mode-card-featured" onClick={() => setMode("bootcamp-setup")}>
                <div className="mode-card-badge">NEW</div>
                <div className="mode-card-icon">🚀</div>
                <h3>AI Interview Bootcamp</h3>
                <p>Get a personalized 1–5 day roadmap. Structured prep with daily goals and unlock system.</p>
                <span className="mode-card-btn featured-btn">Start Bootcamp →</span>
              </div>
            </div>
          </div>
        )}

        {/* PRACTICE — hook only mounts here */}
        {mode === "practice" && (
          <PracticeSession onBack={goHome} />
        )}

        {/* BOOTCAMP SETUP */}
        {mode === "bootcamp-setup" && (
          <BootcampSetup
            onPlanGenerated={handlePlanGenerated}
            onBack={() => setMode("home")}
          />
        )}

        {/* BOOTCAMP ROADMAP */}
        {mode === "bootcamp-roadmap" && bootcampPlan && (
          <BootcampRoadmap
            plan={bootcampPlan}
            completedDays={completedDays}
            onStartDay={handleStartDay}
            onBack={() => { setBootcampPlan(null); setMode("home"); }}
          />
        )}

        {/* BOOTCAMP DAY */}
        {mode === "bootcamp-day" && activeDay && (
          <BootcampDay
            dayItem={activeDay}
            jobRole={bootcampRole}
            onDayComplete={handleDayComplete}
            onBack={() => setMode("bootcamp-roadmap")}
          />
        )}

      </div>
    </>
  );
};

export default App;