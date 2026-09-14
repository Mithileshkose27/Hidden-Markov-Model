import React, { useState, useEffect, useRef } from 'react';
import { Mic, Square, Volume2, Activity, Cpu, GitCommit, RefreshCw, CheckCircle } from 'lucide-react';
import { PRESET_WORDS, generateSyntheticWordAudio, extractWebMFCC } from './utils/webAudio';
import { WebSpeechRecognizer } from './utils/webHmm';

export default function App() {
  const [activeWord, setActiveWord] = useState('zero');
  const [audioSignal, setAudioSignal] = useState(null);
  const [mfccFeatures, setMfccFeatures] = useState([]);
  const [recognitionData, setRecognitionData] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [emIteration, setEmIteration] = useState(1);
  const [logLikelihood, setLogLikelihood] = useState(-42.8);

  const recognizerRef = useRef(null);
  const waveformCanvasRef = useRef(null);
  const mfccCanvasRef = useRef(null);

  // Initialize Recognizer once
  if (!recognizerRef.current) {
    recognizerRef.current = new WebSpeechRecognizer();
  }

  // Process word audio signal
  const processWord = (word) => {
    try {
      setActiveWord(word);
      const signal = generateSyntheticWordAudio(word);
      setAudioSignal(signal);

      const mfcc = extractWebMFCC(signal);
      setMfccFeatures(mfcc || []);

      if (mfcc && mfcc.length > 0 && recognizerRef.current) {
        const res = recognizerRef.current.recognize(mfcc);
        setRecognitionData(res);
      }
    } catch (err) {
      console.error('Error processing word audio:', err);
    }
  };

  // Initial load
  useEffect(() => {
    processWord('zero');
  }, []);

  // Draw Audio Waveform
  useEffect(() => {
    const canvas = waveformCanvasRef.current;
    if (!canvas || !audioSignal || audioSignal.length === 0) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.beginPath();
    const step = Math.max(1, Math.floor(audioSignal.length / width));
    for (let i = 0; i < width; i++) {
      const idx = Math.floor(i * step);
      const val = audioSignal[idx] || 0;
      const y = height / 2 - val * (height / 2.2);
      if (i === 0) ctx.moveTo(i, y);
      else ctx.lineTo(i, y);
    }
    ctx.stroke();
  }, [audioSignal]);

  // Draw MFCC Heatmap
  useEffect(() => {
    const canvas = mfccCanvasRef.current;
    if (!canvas || !mfccFeatures || mfccFeatures.length === 0 || !mfccFeatures[0]) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const T = mfccFeatures.length;
    const numCoeffs = mfccFeatures[0].length;
    const cellW = width / T;
    const cellH = height / numCoeffs;

    for (let t = 0; t < T; t++) {
      for (let c = 0; c < numCoeffs; c++) {
        const val = mfccFeatures[t][c] || 0;
        const norm = Math.max(0, Math.min(1, (val + 5) / 10));
        const hue = (1 - norm) * 240;
        ctx.fillStyle = `hsl(${hue}, 85%, 55%)`;
        ctx.fillRect(t * cellW, (numCoeffs - 1 - c) * cellH, cellW + 0.5, cellH + 0.5);
      }
    }
  }, [mfccFeatures]);

  // Microphone recording
  const startMic = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      const chunks = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        const blob = new Blob(chunks, { type: 'audio/wav' });
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const buf = await blob.arrayBuffer();
        const decoded = await audioCtx.decodeAudioData(buf);
        const floatData = decoded.getChannelData(0);
        setActiveWord('Voice');
        setAudioSignal(floatData);

        const mfcc = extractWebMFCC(floatData);
        setMfccFeatures(mfcc || []);
        if (mfcc && mfcc.length > 0 && recognizerRef.current) {
          const res = recognizerRef.current.recognize(mfcc);
          setRecognitionData(res);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
      setTimeout(() => {
        if (mediaRecorder.state === 'recording') {
          mediaRecorder.stop();
          setIsRecording(false);
          stream.getTracks().forEach((t) => t.stop());
        }
      }, 2500);
    } catch (e) {
      alert('Microphone access unavailable. Using synthetic words.');
    }
  };

  const predictedWord = recognitionData?.predictedWord || 'zero';
  const confidences = recognitionData?.confidences || {};
  const viterbiPath = recognitionData?.bestViterbiPath || [];
  const deltaMatrix = recognitionData?.deltaMatrix || [];

  // Viterbi path state counts
  const stateCounts = [0, 0, 0, 0];
  viterbiPath.forEach((st) => {
    if (st >= 0 && st < 4) stateCounts[st]++;
  });

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div>
          <div className="brand-title">
            <Activity size={26} color="#818cf8" />
            <span>HMM Speech Recognition Studio</span>
          </div>
          <div className="subtitle">
            Continuous Gaussian Hidden Markov Model & Viterbi Trellis Visualizer
          </div>
        </div>
        <span className="badge badge-accent">Interactive HMM Engine</span>
      </header>

      {/* 1. Speech Input Selector */}
      <section className="glass-panel padding-lg margin-bottom-lg">
        <div className="flex-between margin-bottom-md">
          <h2 className="card-title">1. Select Target Speech Word or Record Voice</h2>
          <span className="badge badge-outline">Acoustic Synthesizer</span>
        </div>

        <div className="grid-preset-words margin-bottom-md">
          {PRESET_WORDS.map((w) => (
            <button
              key={w}
              onClick={() => processWord(w)}
              className={`btn btn-preset ${activeWord === w ? 'btn-preset-active' : ''}`}
            >
              <Volume2 size={16} />
              <span>"{w.toUpperCase()}"</span>
            </button>
          ))}
        </div>

        <div>
          <button onClick={startMic} className={`btn ${isRecording ? 'btn-stop animate-pulse' : 'btn-record'}`}>
            <Mic size={18} />
            <span>{isRecording ? 'Recording (2.5s)...' : 'Record Microphone'}</span>
          </button>
        </div>
      </section>

      {/* 2. Audio Waveform & MFCC Heatmap */}
      <section className="glass-panel padding-lg margin-bottom-lg">
        <h2 className="card-title margin-bottom-md">2. Acoustic Signal & 13-Band MFCC Features</h2>
        <div className="grid-spectrogram">
          <div className="spectrogram-box">
            <div className="text-xs text-secondary margin-bottom-xs font-semibold">Time-Domain Audio Waveform</div>
            <canvas ref={waveformCanvasRef} width={600} height={100} className="canvas-styled" />
          </div>
          <div className="spectrogram-box">
            <div className="text-xs text-secondary margin-bottom-xs font-semibold">
              13-Band Mel-Frequency Cepstral Coefficients (MFCC Heatmap)
            </div>
            <canvas ref={mfccCanvasRef} width={600} height={120} className="canvas-styled" />
          </div>
        </div>
      </section>

      {/* 3. Speech Recognition Output */}
      <section className="glass-panel padding-lg margin-bottom-lg">
        <div className="flex-between margin-bottom-md">
          <h2 className="card-title">3. HMM Classifier Recognition Output</h2>
          <span className="badge badge-emerald">Inference Complete</span>
        </div>

        <div className="glass-panel padding-lg margin-bottom-md" style={{ background: 'rgba(99, 102, 241, 0.15)' }}>
          <div className="flex-between flex-wrap gap-md">
            <div>
              <span className="text-xs text-secondary uppercase font-mono">Recognized Word:</span>
              <div className="flex-align gap-sm margin-top-xs">
                <h1 style={{ fontSize: '2.2rem', fontWeight: '800', color: '#ffffff' }}>
                  "{predictedWord.toUpperCase()}"
                </h1>
                <CheckCircle size={26} color="#34d399" />
              </div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <span className="text-xs text-secondary font-mono">Posterior Confidence:</span>
              <div style={{ fontSize: '1.8rem', fontWeight: '800', color: '#38bdf8' }}>
                {((confidences[predictedWord] || 0) * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>

        <div className="confidence-bar-container">
          {PRESET_WORDS.map((w) => {
            const pct = Math.round((confidences[w] || 0) * 100);
            const isWinner = w === predictedWord;
            return (
              <div key={w} className="confidence-row">
                <span className="word-label" style={{ color: isWinner ? '#34d399' : '#94a3b8' }}>
                  "{w.toUpperCase()}"
                </span>
                <div className="bar-track">
                  <div
                    className={`bar-fill ${isWinner ? 'bar-fill-winner' : ''}`}
                    style={{ width: `${Math.max(pct, 4)}%` }}
                  />
                </div>
                <span className="confidence-val">{pct}%</span>
              </div>
            );
          })}
        </div>
      </section>

      {/* 4. Left-to-Right Bakis HMM Topology */}
      <section className="glass-panel padding-lg margin-bottom-lg">
        <h2 className="card-title margin-bottom-md">4. Left-to-Right (Bakis) HMM Topology</h2>
        <div className="hmm-topology-wrapper margin-bottom-md">
          <svg viewBox="0 0 700 160" className="hmm-svg">
            <defs>
              <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#6366f1" />
              </marker>
            </defs>
            <path d="M 120 80 L 230 80" stroke="#6366f1" strokeWidth="2.5" markerEnd="url(#arrow)" />
            <path d="M 290 80 L 400 80" stroke="#6366f1" strokeWidth="2.5" markerEnd="url(#arrow)" />
            <path d="M 460 80 L 570 80" stroke="#6366f1" strokeWidth="2.5" markerEnd="url(#arrow)" />

            {[0, 1, 2, 3].map((idx) => {
              const cx = 90 + idx * 170;
              const cy = 80;
              const isActive = stateCounts[idx] > 0;
              return (
                <g key={idx}>
                  <circle
                    cx={cx}
                    cy={cy}
                    r="28"
                    fill={isActive ? '#0284c7' : '#1e293b'}
                    stroke={isActive ? '#38bdf8' : '#64748b'}
                    strokeWidth={isActive ? '3' : '2'}
                  />
                  <text x={cx} y={cy + 5} fill="#ffffff" fontWeight="bold" fontSize="14" textAnchor="middle">
                    S{idx}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        <div className="grid-state-cards">
          {['S0: Onset', 'S1: Vowel', 'S2: Transition', 'S3: Offset'].map((lbl, i) => (
            <div key={i} className={`state-badge-card ${stateCounts[i] > 0 ? 'state-active-border' : ''}`}>
              <div className="flex-between">
                <span className="state-badge-title">{lbl}</span>
                <span className="state-badge-count">{stateCounts[i]} frames</span>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* 5. Viterbi Trellis Matrix */}
      <section className="glass-panel padding-lg">
        <h2 className="card-title margin-bottom-md">5. Viterbi Dynamic Programming Trellis Matrix</h2>
        <div className="trellis-matrix-container">
          {deltaMatrix && deltaMatrix.length > 0 ? (
            <table className="trellis-table">
              <thead>
                <tr>
                  <th>State \ Time</th>
                  {[0, 4, 8, 12, 16, 20, 24, 28, 32].map((tIdx) => (
                    <th key={tIdx}>t={tIdx + 1}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[0, 1, 2, 3].map((s) => (
                  <tr key={s}>
                    <th>S{s}</th>
                    {[0, 4, 8, 12, 16, 20, 24, 28, 32].map((tIdx) => {
                      const row = deltaMatrix[tIdx];
                      const val = row ? row[s] : -Infinity;
                      const isPath = viterbiPath[tIdx] === s;
                      return (
                        <td key={tIdx} className={isPath ? 'cell-viterbi-active' : ''}>
                          {val === -Infinity ? '-∞' : val.toFixed(1)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-secondary text-sm">Matrix populates upon word selection.</p>
          )}
        </div>
      </section>
    </div>
  );
}
