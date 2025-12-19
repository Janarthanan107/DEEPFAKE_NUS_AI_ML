import { useState } from 'react'
import './App.css'

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [dragActive, setDragActive] = useState(false)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = (file) => {
    if (file.type.startsWith('video/')) {
      setFile(file)
      setPreview(URL.createObjectURL(file))
      setResult(null)
    } else {
      alert("Please upload a video file.")
    }
  }

  const analyzeVideo = async () => {
    if (!file) return
    setLoading(true)
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      // Assuming API is running on localhost:8000
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()
      setResult(data)
    } catch (error) {
      console.error(error)
      setResult({ error: "Failed to connect to API" })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <div className="glass-card">
        <h1>Deepfake Detector</h1>
        <p className="subtitle">AI-Powered Forensic Analysis</p>

        <div
          className={`upload-area ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => document.getElementById('file-input').click()}
        >
          <span className="upload-icon">📹</span>
          <p>{file ? file.name : "Drag & Drop video here or Click to Browse"}</p>
          <input
            type="file"
            id="file-input"
            accept="video/*"
            onChange={handleChange}
            hidden
          />
        </div>

        {preview && (
          <video className="preview-video" src={preview} controls />
        )}

        <button
          className="btn-primary"
          onClick={analyzeVideo}
          disabled={!file || loading}
        >
          {loading ? "Analyzing..." : "Analyze Video"}
        </button>

        {loading && <span className="loader"></span>}

        {result && (
          <div className={`result-container ${result.final_prediction === 'FAKE' ? 'result-fake' : result.final_prediction === 'REAL' ? 'result-real' : ''}`}>
            {result.error ? (
              <h3>Error: {result.error}</h3>
            ) : (
              <>
                <h2>Verdict: {result.final_prediction}</h2>
                <p>Confidence: {result.confidence ? (result.confidence * 100).toFixed(2) : 0}%</p>
                <div style={{ textAlign: 'left', marginTop: '1rem', fontSize: '0.9rem' }}>
                  <p><strong>Method Used:</strong> {result.routing}</p>
                  {result.details?.ViT && <p>📸 ViT Score: {result.details.ViT.score.toFixed(4)}</p>}
                  {result.details?.CNN && <p>🎥 CNN Score: {result.details.CNN.score.toFixed(4)}</p>}
                  {!result.confidence && <p>(No models returned a valid score)</p>}
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
