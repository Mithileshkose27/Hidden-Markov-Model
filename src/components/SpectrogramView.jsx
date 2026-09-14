import React, { useEffect, useRef } from 'react';
import { Waves, Grid, Cpu } from 'lucide-react';

export default function SpectrogramView({ audioSignal, mfccFeatures }) {
  const waveformCanvasRef = useRef(null);
  const mfccCanvasRef = useRef(null);

  // Render Audio Waveform
  useEffect(() => {
    const canvas = waveformCanvasRef.current;
    if (!canvas || !audioSignal || audioSignal.length === 0) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    // Center Line
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Waveform Curve
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.beginPath();

    const step = Math.max(1, Math.floor(audioSignal.length / width));
    const amp = height / 2.2;

    for (let i = 0; i < width; i++) {
      const idx = Math.floor(i * step);
      const val = audioSignal[idx] || 0;
      const y = height / 2 - val * amp;

      if (i === 0) ctx.moveTo(i, y);
      else ctx.lineTo(i, y);
    }
    ctx.stroke();
  }, [audioSignal]);

  // Render 13-band MFCC Feature Heatmap
  useEffect(() => {
    const canvas = mfccCanvasRef.current;
    if (!canvas || !mfccFeatures || !Array.isArray(mfccFeatures) || mfccFeatures.length === 0 || !mfccFeatures[0]) return;

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
        const val = mfccFeatures[t][c];
        // Heatmap HSL mapping
        const norm = Math.max(0, Math.min(1, (val + 5) / 10));
        const hue = (1 - norm) * 240; // Blue (cold) to Red (hot)
        ctx.fillStyle = `hsl(${hue}, 85%, 55%)`;
        ctx.fillRect(t * cellW, (numCoeffs - 1 - c) * cellH, cellW + 0.5, cellH + 0.5);
      }
    }
  }, [mfccFeatures]);

  return (
    <div className="card glass-panel padding-lg">
      <div className="flex-between margin-bottom-md">
        <div className="flex-align gap-sm">
          <Waves className="icon-primary" size={22} />
          <h2 className="card-title">2. Acoustic Feature Extraction (MFCC Heatmap)</h2>
        </div>
        <span className="badge badge-outline">Signal Processing</span>
      </div>

      <div className="grid-spectrogram">
        {/* Waveform View */}
        <div className="spectrogram-box">
          <div className="flex-between margin-bottom-xs">
            <span className="text-xs text-secondary flex-align gap-xs">
              <ActivityIcon size={14} /> Time-Domain Waveform
            </span>
            <span className="text-xs text-muted">Amplitude vs Time</span>
          </div>
          <canvas ref={waveformCanvasRef} width={600} height={120} className="canvas-styled" />
        </div>

        {/* MFCC Feature Heatmap */}
        <div className="spectrogram-box">
          <div className="flex-between margin-bottom-xs">
            <span className="text-xs text-secondary flex-align gap-xs">
              <Cpu size={14} /> 13-Band Mel-Frequency Cepstral Coefficients (MFCC)
            </span>
            <span className="text-xs text-muted">{mfccFeatures.length} Frames × 13 Coeffs</span>
          </div>
          <canvas ref={mfccCanvasRef} width={600} height={140} className="canvas-styled" />
        </div>
      </div>
    </div>
  );
}

function ActivityIcon({ size }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
    </svg>
  );
}
