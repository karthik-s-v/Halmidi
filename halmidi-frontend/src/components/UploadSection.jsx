import { useState } from "react";
import "./UploadSection.css";

function UploadSection({ onUpload, onTranslate }) {
  const [preview, setPreview] = useState(null);

  const handleChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setPreview(URL.createObjectURL(file));
      onUpload(e);
    }
  };

  return (
    <div className="card upload-card">
      <h3>Step 1: Upload Inscription Image</h3>

      <label className="upload-box">
        <span className="upload-title">
          Click to upload or drag & drop
        </span> 
        <br></br>
        <span className="upload-sub">
          JPG / PNG up to 10MB
        </span>

        <input
          type="file"
          accept="image/*"
          hidden
          onChange={handleChange}
        />
      </label>

      {preview && (
        <div className="preview-box">
          <img src={preview} alt="Uploaded inscription" />
        </div>
      )}

      <button className="translate-btn" onClick={onTranslate}>
        Translate
      </button>

      <small className="tip">
        TIP: Ensure high contrast script background.
      </small>
    </div>
  );
}

export default UploadSection;
