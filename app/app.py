import { useRef, useState } from "react";
import "./App.css";


const API_URL =
  import.meta.env.VITE_API_URL ||
  "http://localhost:8000";


const MODELS = {
  rail: {
    name: "Rail Fault",
    endpoint: "/rail/predict",
    description:
      "Upload a rail signal CSV for fault prediction.",
  },

  door: {
    name: "Door Fault",
    endpoint: "/door/predict",
    description:
      "Upload a door signal CSV for fault prediction.",
  },

  shm: {
    name: "Structural Health Monitoring",
    endpoint: "/shm/predict",
    description:
      "Upload an SHM CSV for condition prediction.",
  },

  acv: {
    name: "ACV",
    endpoint: "/acv/predict",
    description:
      "Upload an ACV CSV for prediction.",
  },
};


export default function App() {

  const [selectedModel, setSelectedModel] =
    useState(null);

  const [selectedFile, setSelectedFile] =
    useState(null);

  const [prediction, setPrediction] =
    useState(null);

  const [confidence, setConfidence] =
    useState(null);

  const [probabilities, setProbabilities] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState(null);

  const [success, setSuccess] =
    useState(null);

  const fileInputRef = useRef(null);


  // ==========================================================
  // SELECT MODEL
  // ==========================================================

  function selectModel(modelKey) {

    setSelectedModel(modelKey);

    setSelectedFile(null);

    setPrediction(null);

    setConfidence(null);

    setProbabilities(null);

    setError(null);

    setSuccess(null);

    if (fileInputRef.current) {

      fileInputRef.current.value = "";
    }
  }


  // ==========================================================
  // FILE SELECTED
  // ==========================================================

  function handleFileChange(event) {

    const file =
      event.target.files?.[0];

    if (!file) {
      return;
    }

    setSelectedFile(file);

    setPrediction(null);

    setConfidence(null);

    setProbabilities(null);

    setError(null);

    setSuccess(null);
  }


  // ==========================================================
  // UPLOAD
  // ==========================================================

  async function runPrediction() {

    if (!selectedModel) {

      setError(
        "Please select a model first."
      );

      return;
    }

    if (!selectedFile) {

      setError(
        "Please upload a CSV file first."
      );

      return;
    }


    const model =
      MODELS[selectedModel];


    setLoading(true);

    setError(null);

    setPrediction(null);

    setConfidence(null);

    setProbabilities(null);

    setSuccess(null);


    try {

      const formData =
        new FormData();

      formData.append(
        "file",
        selectedFile
      );


      const response =
        await fetch(
          `${API_URL}${model.endpoint}`,
          {
            method: "POST",
            body: formData,
          }
        );


      const data =
        await response.json();


      if (!response.ok) {

        throw new Error(
          data.detail ||
          "Prediction failed."
        );
      }


      setPrediction(
        data.prediction
      );


      if (
        data.confidence !== undefined
      ) {

        setConfidence(
          data.confidence
        );
      }


      if (
        data.probabilities
      ) {

        setProbabilities(
          data.probabilities
        );
      }


      setSuccess(
        `Prediction completed for ${selectedFile.name}`
      );

    } catch (err) {

      setError(
        err.message ||
        "Could not connect to the API."
      );

    } finally {

      setLoading(false);
    }
  }


  // ==========================================================
  // RESET
  // ==========================================================

  function reset() {

    setSelectedFile(null);

    setPrediction(null);

    setConfidence(null);

    setProbabilities(null);

    setError(null);

    setSuccess(null);

    if (fileInputRef.current) {

      fileInputRef.current.value = "";
    }
  }


  // ==========================================================
  // UI
  // ==========================================================

  return (

    <div className="app">

      <header className="header">

        <div>

          <h1>
            Fault Prediction Dashboard
          </h1>

          <p>
            Select a model, upload your CSV,
            and run a prediction.
          </p>

        </div>

      </header>


      <main className="container">


        {/* =================================================
            MODEL BUTTONS
        ================================================== */}

        <section className="model-section">

          <h2>
            Select Model
          </h2>


          <div className="model-grid">

            {Object.entries(MODELS).map(
              ([key, model]) => (

                <button
                  key={key}
                  className={
                    `model-button ${
                      selectedModel === key
                        ? "active"
                        : ""
                    }`
                  }
                  onClick={() =>
                    selectModel(key)
                  }
                >

                  <span className="model-title">
                    {model.name}
                  </span>

                  <span className="model-description">
                    {model.description}
                  </span>

                </button>

              )
            )}

          </div>

        </section>


        {/* =================================================
            UPLOAD AREA
        ================================================== */}

        {selectedModel && (

          <section className="upload-section">

            <h2>
              {MODELS[selectedModel].name}
            </h2>


            <div className="upload-box">

              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,text/csv"
                onChange={
                  handleFileChange
                }
              />


              {selectedFile && (

                <div className="file-info">

                  <strong>
                    Selected file:
                  </strong>

                  <span>
                    {selectedFile.name}
                  </span>

                  <span>
                    {(
                      selectedFile.size /
                      1024
                    ).toFixed(1)}
                    {" KB"}
                  </span>

                </div>

              )}

            </div>


            <div className="actions">

              <button
                className="predict-button"
                disabled={
                  loading ||
                  !selectedFile
                }
                onClick={
                  runPrediction
                }
              >

                {loading
                  ? "Running model..."
                  : `Run ${MODELS[selectedModel].name}`
                }

              </button>


              <button
                className="reset-button"
                onClick={reset}
                disabled={loading}
              >
                Reset
              </button>

            </div>

          </section>

        )}


        {/* =================================================
            ERROR
        ================================================== */}

        {error && (

          <div className="message error">

            <strong>
              Error
            </strong>

            <p>
              {error}
            </p>

          </div>

        )}


        {/* =================================================
            SUCCESS
        ================================================== */}

        {success && (

          <div className="message success">

            {success}

          </div>

        )}


        {/* =================================================
            RESULT
        ================================================== */}

        {prediction !== null && (

          <section className="result-section">

            <h2>
              Prediction Result
            </h2>


            <div className="prediction-card">

              <span className="result-label">
                Model
              </span>

              <strong>
                {MODELS[selectedModel].name}
              </strong>


              <span className="result-label">
                Prediction
              </span>

              <div className="prediction">

                {String(prediction)}

              </div>


              {confidence !== null && (

                <>

                  <span className="result-label">
                    Confidence
                  </span>

                  <div className="confidence">

                    {(confidence * 100).toFixed(2)}
                    %

                  </div>

                </>

              )}

            </div>


            {/* =============================================
                CLASS PROBABILITIES
            ============================================== */}

            {probabilities && (

              <div className="probabilities">

                <h3>
                  Class Probabilities
                </h3>


                {probabilities.map(
                  (probability, index) => (

                    <div
                      className="probability-row"
                      key={index}
                    >

                      <span>
                        Class {index + 1}
                      </span>

                      <div className="bar">

                        <div
                          className="bar-fill"
                          style={{
                            width:
                              `${probability * 100}%`,
                          }}
                        />

                      </div>

                      <span>
                        {(
                          probability *
                          100
                        ).toFixed(2)}
                        %
                      </span>

                    </div>

                  )
                )}

              </div>

            )}

          </section>

        )}

      </main>

    </div>
  );
}