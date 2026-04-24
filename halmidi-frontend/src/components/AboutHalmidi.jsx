import "./AboutHalmidi.css";

function AboutHalmidi() {
  return (
    <div className="sidebar">
      <h2>Halmidi Shasana</h2>

      <p>
        Halmidi Shasana (450 AD) is the earliest known Kannada inscription,
        discovered in Halmidi village, Hassan district.
      </p>

      <p>
        It marks the transition of Kannada as an administrative language
        under the Kadamba dynasty.
      </p>

      <p>
        This project uses AI-based OCR and Gemini Multimodal AI to analyze
        and translate ancient Kannada inscriptions.
      </p>

      {/* ✅ REPLACED CREATED BY SECTION */}
      <div className="credits-section">
        <h3 className="credits-title">Project Team</h3>

        <div className="credit-card">
          <span className="credit-name">Karthik Kumar S V</span>
        </div>

        <div className="credit-card">
          <span className="credit-name">Kushaal B</span>
        </div>

        <div className="credit-card">
          <span className="credit-name">Prekshitha K R</span>
        </div>

        <div className="credit-card">
          <span className="credit-name">Ponnamma K M</span>
        </div>
      </div>
    </div>
  );
}

export default AboutHalmidi;
