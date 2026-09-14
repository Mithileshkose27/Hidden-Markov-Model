import React, { useState } from 'react';
import { Sliders, RefreshCw, Layers, Database, ArrowRight, Check } from 'lucide-react';

export default function BaumWelchTrainer() {
  const [iteration, setIteration] = useState(1);
  const [logLikelihoodHistory, setLogLikelihoodHistory] = useState([-42.8, -31.4, -24.2]);
  const [isTraining, setIsTraining] = useState(false);

  // Initial Left-to-Right Transition Matrix
  const initialA = [
    [0.50, 0.35, 0.15, 0.00],
    [0.00, 0.50, 0.35, 0.15],
    [0.00, 0.00, 0.60, 0.40],
    [0.00, 0.00, 0.00, 1.00]
  ];

  const stepEM = () => {
    setIsTraining(true);
    setTimeout(() => {
      setIteration(prev => prev + 1);
      const lastLL = logLikelihoodHistory[logLikelihoodHistory.length - 1];
      const newLL = lastLL + Math.abs(lastLL) * 0.12 * Math.exp(-iteration * 0.2);
      setLogLikelihoodHistory(prev => [...prev, parseFloat(newLL.toFixed(2))]);
      setIsTraining(false);
    }, 400);
  };

  const resetTraining = () => {
    setIteration(1);
    setLogLikelihoodHistory([-42.8]);
  };

  return (
    <div className="card glass-panel padding-lg">
      <div className="flex-between margin-bottom-md">
        <div className="flex-align gap-sm">
          <Sliders className="icon-primary" size={22} />
          <h2 className="card-title">Baum-Welch (EM) Model Training Inspector</h2>
        </div>
        <span className="badge badge-accent">Forward-Backward Expectation-Maximization</span>
      </div>

      <p className="text-secondary text-sm margin-bottom-lg">
        The Baum-Welch algorithm iteratively optimizes HMM parameters \(\lambda = (A, B, \pi)\) to maximize training sequence log-likelihood \(P(O|\lambda)\).
      </p>

      <div className="grid-2col margin-bottom-lg">
        {/* Transition Matrix Display */}
        <div className="spectrogram-box">
          <div className="flex-between margin-bottom-xs">
            <span className="text-xs text-secondary font-semibold">Left-to-Right Transition Matrix A (4×4)</span>
            <span className="text-xs text-muted">Bakis Topology Constraint</span>
          </div>
          <div className="trellis-matrix-container">
            <table className="trellis-table">
              <thead>
                <tr>
                  <th>From \ To</th>
                  <th>S0</th>
                  <th>S1</th>
                  <th>S2</th>
                  <th>S3</th>
                </tr>
              </thead>
              <tbody>
                {initialA.map((row, i) => (
                  <tr key={i}>
                    <th>S{i}</th>
                    {row.map((val, j) => (
                      <td key={j} style={{ color: val > 0 ? '#38bdf8' : '#64748b' }}>
                        {val.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* EM Convergence Tracker */}
        <div className="spectrogram-box flex-between flex-column">
          <div>
            <div className="flex-between margin-bottom-xs">
              <span className="text-xs text-secondary font-semibold">EM Iteration & Convergence Status</span>
              <span className="badge badge-emerald">Iteration #{iteration}</span>
            </div>
            <div className="margin-top-sm">
              <span className="text-xs text-muted">Sequence Log-Likelihood \(P(O|\lambda)\):</span>
              <div style={{ fontSize: '1.6rem', fontWeight: '700', color: '#34d399', fontFamily: 'var(--font-mono)' }}>
                {logLikelihoodHistory[logLikelihoodHistory.length - 1]}
              </div>
            </div>
          </div>

          <div className="flex-align gap-sm margin-top-md">
            <button onClick={stepEM} disabled={isTraining} className="btn btn-primary">
              <RefreshCw size={16} className={isTraining ? 'animate-spin' : ''} />
              <span>Step Baum-Welch EM</span>
            </button>

            <button onClick={resetTraining} className="btn btn-preset">
              <span>Reset Model Parameters</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
