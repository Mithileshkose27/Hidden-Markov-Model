import React from 'react';
import { Grid, HelpCircle, Layers, ArrowUpRight } from 'lucide-react';

export default function ViterbiTrellis({ deltaMatrix, viterbiPath, wordScore }) {
  if (!deltaMatrix || !Array.isArray(deltaMatrix) || deltaMatrix.length === 0 || !deltaMatrix[0]) {
    return (
      <div className="card glass-panel padding-lg">
        <h2 className="card-title margin-bottom-sm">Viterbi Dynamic Programming Trellis</h2>
        <p className="text-secondary text-sm">Select or record an audio sample above to compute Viterbi trellis dynamic programming matrix.</p>
      </div>
    );
  }

  const T = deltaMatrix.length;
  const nStates = deltaMatrix[0].length;
  // Display sub-sample of max 16 frames for layout clarity if audio is long
  const stride = Math.max(1, Math.ceil(T / 16));
  const sampledIndices = [];
  for (let t = 0; t < T; t += stride) {
    sampledIndices.push(t);
  }

  return (
    <div className="card glass-panel padding-lg">
      <div className="flex-between margin-bottom-md">
        <div className="flex-align gap-sm">
          <Grid className="icon-primary" size={22} />
          <h2 className="card-title">4. Viterbi Dynamic Programming Trellis Matrix (δ_t(i))</h2>
        </div>
        <span className="badge badge-accent">DP Complexity O(N² T)</span>
      </div>

      <p className="text-secondary text-sm margin-bottom-lg">
        The Viterbi algorithm computes maximum logarithmic path likelihoods across state transitions:
        δ_t(j) = max_i [δ_t-1(i) + ln(a_ij)] + ln(b_j(o_t)). Highlighted cells mark the optimal state sequence Q*.
      </p>

      {/* Dynamic Programming Matrix Table */}
      <div className="trellis-matrix-container margin-bottom-md">
        <table className="trellis-table">
          <thead>
            <tr>
              <th>State \ Time</th>
              {sampledIndices.map((tIdx) => (
                <th key={tIdx}>t = {tIdx + 1}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[0, 1, 2, 3].map((stateIdx) => (
              <tr key={stateIdx}>
                <th>S{stateIdx}</th>
                {sampledIndices.map((tIdx) => {
                  const val = deltaMatrix[tIdx][stateIdx];
                  const isViterbiPath = viterbiPath && viterbiPath[tIdx] === stateIdx;
                  return (
                    <td
                      key={tIdx}
                      className={isViterbiPath ? 'cell-viterbi-active' : ''}
                      title={`Frame ${tIdx + 1}, State ${stateIdx}: log-prob ${val.toFixed(2)}`}
                    >
                      <span className="cell-score">
                        {val === -Infinity ? '-∞' : val.toFixed(1)}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex-between flex-wrap gap-md margin-top-md">
        <div className="flex-align gap-sm text-sm">
          <span className="badge badge-emerald">Optimal Path Highlighted</span>
          <span className="text-secondary">Normalised Log-Likelihood: <strong>{wordScore ? wordScore.toFixed(3) : 'N/A'}</strong> per frame</span>
        </div>
        <div className="text-xs text-muted">
          *Showing {sampledIndices.length} representative acoustic frames out of {T} total frames.
        </div>
      </div>
    </div>
  );
}
