import React from 'react';
import { GitCommit, ArrowRight, RefreshCw, Layers } from 'lucide-react';

export default function HmmGraph({ activePath = [], nStates = 4 }) {
  // Count frequency of states visited in Viterbi path
  const stateCounts = new Array(nStates).fill(0);
  const safePath = Array.isArray(activePath) ? activePath : [];
  safePath.forEach((st) => {
    if (st >= 0 && st < nStates) stateCounts[st]++;
  });

  const stateLabels = [
    { name: 'S₀: Silence/Onset', desc: 'Acoustic Onset phase' },
    { name: 'S₁: Vowel Nucleus', desc: 'Main resonant vowel formant' },
    { name: 'S₂: Transition', desc: 'Inter-phoneme trajectory' },
    { name: 'S₃: Coda / Offset', desc: 'Consonant closure / silence' }
  ];

  return (
    <div className="card glass-panel padding-lg">
      <div className="flex-between margin-bottom-md">
        <div className="flex-align gap-sm">
          <GitCommit className="icon-primary" size={22} />
          <h2 className="card-title">3. Left-to-Right (Bakis) HMM Topology</h2>
        </div>
        <span className="badge badge-accent">Phoneme State Graph</span>
      </div>

      <p className="text-secondary text-sm margin-bottom-md">
        In speech recognition, HMM hidden states represent temporal acoustic phases. Transitions are strictly Left-to-Right ($a_{i,j} = 0$ for $j &lt; i$) to match time's forward arrow.
      </p>

      {/* SVG Topology Graph */}
      <div className="hmm-topology-wrapper">
        <svg viewBox="0 0 700 180" className="hmm-svg">
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#6366f1" />
            </marker>
            <marker id="arrow-active" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#38bdf8" />
            </marker>
          </defs>

          {/* Forward Transitions */}
          {/* S0 -> S1 */}
          <path d="M 120 90 L 230 90" stroke="#6366f1" strokeWidth="2.5" markerEnd="url(#arrow)" />
          <text x="175" y="80" fill="#a5b4fc" fontSize="12" textAnchor="middle">a₀₁ = 0.35</text>

          {/* S1 -> S2 */}
          <path d="M 290 90 L 400 90" stroke="#6366f1" strokeWidth="2.5" markerEnd="url(#arrow)" />
          <text x="345" y="80" fill="#a5b4fc" fontSize="12" textAnchor="middle">a₁₂ = 0.35</text>

          {/* S2 -> S3 */}
          <path d="M 460 90 L 570 90" stroke="#6366f1" strokeWidth="2.5" markerEnd="url(#arrow)" />
          <text x="515" y="80" fill="#a5b4fc" fontSize="12" textAnchor="middle">a₂₃ = 0.40</text>

          {/* Skip Transitions (Arc S0 -> S2) */}
          <path d="M 100 65 Q 265 15 420 65" fill="none" stroke="#818cf8" strokeWidth="1.8" strokeDasharray="4 4" markerEnd="url(#arrow)" />
          <text x="265" y="32" fill="#c7d2fe" fontSize="11" textAnchor="middle">Skip: a₀₂ = 0.15</text>

          {/* State Nodes */}
          {[0, 1, 2, 3].map((idx) => {
            const cx = 90 + idx * 170;
            const cy = 90;
            const isActive = stateCounts[idx] > 0;

            return (
              <g key={idx}>
                {/* Self-loop arc */}
                <path
                  d={`M ${cx - 15} ${cy - 20} A 20 20 0 1 1 ${cx + 15} ${cy - 20}`}
                  fill="none"
                  stroke={isActive ? '#38bdf8' : '#475569'}
                  strokeWidth="2"
                  markerEnd="url(#arrow)"
                />
                <text x={cx} y={cy - 50} fill="#94a3b8" fontSize="11" textAnchor="middle">
                  a{idx}{idx} = 0.50
                </text>

                {/* State Circle */}
                <circle
                  cx={cx}
                  cy={cy}
                  r="30"
                  fill={isActive ? 'url(#activeGlow)' : '#1e293b'}
                  stroke={isActive ? '#38bdf8' : '#64748b'}
                  strokeWidth={isActive ? '3' : '2'}
                  className={isActive ? 'state-circle-active' : ''}
                />
                <text x={cx} y={cy + 5} fill="#f8fafc" fontWeight="bold" fontSize="14" textAnchor="middle">
                  S{idx}
                </text>
              </g>
            );
          })}

          <linearGradient id="activeGlow" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#0284c7" />
            <stop offset="100%" stopColor="#4f46e5" />
          </linearGradient>
        </svg>
      </div>

      {/* State Breakdown Badges */}
      <div className="grid-state-cards margin-top-md">
        {stateLabels.map((st, i) => (
          <div key={i} className={`state-badge-card ${stateCounts[i] > 0 ? 'state-active-border' : ''}`}>
            <div className="flex-between">
              <span className="state-badge-title">{st.name}</span>
              <span className="state-badge-count">{stateCounts[i]} frames</span>
            </div>
            <p className="state-badge-desc">{st.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
