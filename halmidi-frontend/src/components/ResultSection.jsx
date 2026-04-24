import "./ResultSection.css";

function ResultSection({ text, analysis }) {
  return (
    <div className="card results-card">
      <h3>Step 2: Analysis Results</h3>

      {!text ? (
        <p className="placeholder">
          Upload an inscription and click <b>Translate</b> to view results.
        </p>
      ) : (
        <>
          <div className="result-block">
            <h4>Recognized Modern Kannada Text</h4>
            <div className="kannada-box">{text}</div>
          </div>

          <div className="result-block">
            <h4>Scholarly Interpretation</h4>
            <p className="analysis-text">{analysis}</p>
          </div>
        </>
      )}
    </div>
  );
}

export default ResultSection;
