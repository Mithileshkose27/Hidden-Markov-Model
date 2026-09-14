import React from 'react';
import { Award, BarChart3, CheckCircle2, Flame, Zap } from 'lucide-react';

export default function RecognitionResult({ recognitionData, activeWord }) {
  if (!recognitionData || !recognitionData.predictedWord || !recognitionData.confidences || !recognitionData.results) {
    return (
      <div className="card glass-panel padding-lg">
        <h2 className="card-title margin-bottom-sm">Recognition Output & Probabilities</h2>
        <p className="text-secondary text-sm">Waiting for audio signal analysis...</p>
      </div>
    );
  }

  const { predictedWord, confidences, results } = recognitionData;
  const isMatch = activeWord && activeWord.toLowerCase() === predictedWord.toLowerCase();

  return (
    <div className="card glass-panel padding-lg">
      <div className="flex-between margin-bottom-md">
        <div className="flex-align gap-sm">
          <Award className="icon-primary" size={22} />
          <h2 className="card-title">5. HMM Speech Recognition Output</h2>
        </div>
        <span className="badge badge-emerald">Inference Complete</span>
      </div>

      {/* Top Prediction Display Banner */}
      <div
        className="glass-panel padding-lg margin-bottom-lg"
        style={{
          background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(56, 189, 248, 0.15) 100%)',
          borderColor: 'rgba(99, 102, 241, 0.4)'
        }}
      >
        <div className="flex-between flex-wrap gap-md">
          <div>
            <span className="text-xs text-secondary uppercase font-mono tracking-wider">Top Recognized Acoustic Word:</span>
            <div className="flex-align gap-sm margin-top-xs">
              <h1 style={{ fontSize: '2.4rem', fontWeight: '800', letterSpacing: '1px', color: '#ffffff' }}>
                "{predictedWord.toUpperCase()}"
              </h1>
              <CheckCircle2 size={28} color="#34d399" />
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

      {/* Vocabulary Probability Distribution Bar Charts */}
      <h3 className="text-sm font-semibold margin-bottom-md text-secondary flex-align gap-xs">
        <BarChart3 size={16} /> Softmax Log-Likelihood Confidence Distribution Across Vocabulary:
      </h3>

      <div className="confidence-bar-container">
        {Object.keys(confidences).map((w) => {
          const confPct = Math.round(confidences[w] * 100);
          const isWinner = w === predictedWord;
          const score = results[w] ? results[w].score.toFixed(2) : 'N/A';

          return (
            <div key={w} className="confidence-row">
              <span className="word-label" style={{ color: isWinner ? '#34d399' : '#94a3b8' }}>
                "{w.toUpperCase()}"
              </span>
              <div className="bar-track">
                <div
                  className={`bar-fill ${isWinner ? 'bar-fill-winner' : ''}`}
                  style={{ width: `${Math.max(confPct, 4)}%` }}
                />
              </div>
              <span className="confidence-val">
                {confPct}% <span className="text-xs text-muted">({score})</span>
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
