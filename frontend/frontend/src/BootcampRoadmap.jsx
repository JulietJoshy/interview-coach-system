const dayIcons = ["💬", "🧠", "🛠️", "⚙️", "🎯"];

const BootcampRoadmap = ({ plan, completedDays, onStartDay, onBack }) => {
    const { role, total_days, plan: days } = plan;

    return (
        <div className="bootcamp-roadmap-page">
            <div className="roadmap-header">
                <button className="back-btn" onClick={onBack}>
                    ← Back
                </button>
                <div className="roadmap-title-block">
                    <h1>Your {total_days}-Day Bootcamp</h1>
                    <p className="hero-subtitle" style={{ margin: 0 }}>
                        Role: <strong style={{ color: "#3b82f6" }}>{role}</strong> &nbsp;·&nbsp;
                        {completedDays.length}/{total_days} days completed
                    </p>
                </div>
                {/* Overall progress bar */}
                <div className="roadmap-progress-bar">
                    <div
                        className="roadmap-progress-fill"
                        style={{ width: `${(completedDays.length / total_days) * 100}%` }}
                    />
                </div>
            </div>

            <div className="roadmap-timeline">
                {days.map((dayItem, idx) => {
                    const dayNum = dayItem.day;
                    const isCompleted = completedDays.includes(dayNum);
                    const isActive = dayNum === 1 || completedDays.includes(dayNum - 1);
                    const isLocked = !isCompleted && !isActive;
                    const isCurrent = isActive && !isCompleted;

                    return (
                        <div
                            key={dayNum}
                            className={`roadmap-day-card ${isCompleted ? "day-done" : ""} ${isCurrent ? "day-current" : ""} ${isLocked ? "day-locked" : ""}`}
                        >
                            {/* Connector line */}
                            {idx < days.length - 1 && (
                                <div className={`timeline-connector ${isCompleted ? "connector-done" : ""}`} />
                            )}

                            {/* Day Status Circle */}
                            <div className={`day-status-circle ${isCompleted ? "circle-done" : isCurrent ? "circle-current" : "circle-locked"}`}>
                                {isCompleted ? "✓" : dayIcons[idx]}
                            </div>

                            {/* Day Info */}
                            <div className="day-info">
                                <div className="day-info-header">
                                    <span className="day-number-tag">Day {dayNum}</span>
                                    {isCompleted && <span className="day-badge done-badge">✅ Complete</span>}
                                    {isCurrent && <span className="day-badge current-badge">● Today</span>}
                                    {isLocked && <span className="day-badge locked-badge">🔒 Locked</span>}
                                </div>
                                <h3 className="day-card-title">{dayItem.title}</h3>
                                <p className="day-card-focus">{dayItem.focus}</p>
                                <p className="day-card-questions">
                                    {dayItem.questions.length} questions
                                </p>
                            </div>

                            {/* Action Button */}
                            <div className="day-action">
                                {isCompleted && (
                                    <button
                                        className="day-action-btn retry-btn"
                                        onClick={() => onStartDay(dayItem)}
                                    >
                                        Retry
                                    </button>
                                )}
                                {isCurrent && (
                                    <button
                                        className="day-action-btn start-day-btn"
                                        onClick={() => onStartDay(dayItem)}
                                    >
                                        Start →
                                    </button>
                                )}
                                {isLocked && (
                                    <button className="day-action-btn locked-btn" disabled>
                                        Locked
                                    </button>
                                )}
                            </div>
                        </div>
                    );
                })}
            </div>

            {completedDays.length === total_days && (
                <div className="bootcamp-complete-banner fade-in">
                    <h2>🎉 Bootcamp Complete!</h2>
                    <p>You've finished all {total_days} days of preparation. Good luck with your interview!</p>
                    <button className="begin-btn" style={{ maxWidth: "280px", margin: "0 auto" }} onClick={onBack}>
                        Start a New Bootcamp
                    </button>
                </div>
            )}
        </div>
    );
};

export default BootcampRoadmap;
