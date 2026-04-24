import { useState } from "react";
import axios from "axios";

import Header from "./components/Header";
import UploadSection from "./components/UploadSection";
import ResultSection from "./components/ResultSection";
import AboutHalmidi from "./components/AboutHalmidi";

import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [text, setText] = useState("");
  const [analysis, setAnalysis] = useState("");

  const handleUpload = (e) => {
    setFile(e.target.files[0]);
  };

  const handleTranslate = async () => {
    if (!file) {
      alert("Please upload an image first");
      return;
    }

    const formData = new FormData();
    formData.append("image", file);

    const res = await axios.post(
      "http://127.0.0.1:5000/predict",
      formData
    );

    setText(res.data.kannada_text);
    setAnalysis(res.data.analysis);
  };

  return (
    <div className="app">
      <AboutHalmidi />

      <div className="main">
        <Header />

        <div className="grid">
          <UploadSection
            onUpload={handleUpload}
            onTranslate={handleTranslate}
          />
          <ResultSection
            text={text}
            analysis={analysis}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
