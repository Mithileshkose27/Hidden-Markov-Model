/**
 * JavaScript HMM Engine: Continuous Gaussian Emission HMM & Viterbi Trellis DP
 */

export class WebGaussianHMM {
  constructor(word, nStates = 4, nFeatures = 13) {
    this.word = word;
    this.nStates = nStates;
    this.nFeatures = nFeatures;
    
    // Bakis Left-to-Right Initial probabilities pi
    this.pi = new Array(nStates).fill(-Infinity);
    this.pi[0] = 0.0; // log(1.0) = 0.0

    // Left-to-Right Bakis Transition Matrix logA
    this.logA = Array.from({ length: nStates }, () => new Array(nStates).fill(-Infinity));
    for (let i = 0; i < nStates; i++) {
      if (i === nStates - 1) {
        this.logA[i][i] = Math.log(1.0);
      } else if (i === nStates - 2) {
        this.logA[i][i] = Math.log(0.6);
        this.logA[i][i + 1] = Math.log(0.4);
      } else {
        this.logA[i][i] = Math.log(0.5);
        this.logA[i][i + 1] = Math.log(0.35);
        this.logA[i][i + 2] = Math.log(0.15);
      }
    }

    // Means & Covariances per state
    this.means = Array.from({ length: nStates }, (_, s) => 
      Array.from({ length: nFeatures }, (_, f) => Math.sin(s * 1.5 + f * 0.8) * 2.0)
    );
    this.covars = Array.from({ length: nStates }, () => 
      new Array(nFeatures).fill(1.5)
    );
  }

  computeLogEmission(frame) {
    const logB = new Array(this.nStates);
    for (let j = 0; j < this.nStates; j++) {
      let mah = 0;
      let logDet = 0;
      for (let d = 0; d < this.nFeatures; d++) {
        const diff = frame[d] - this.means[j][d];
        const v = Math.max(this.covars[j][d], 0.1);
        mah += (diff * diff) / v;
        logDet += Math.log(2 * Math.PI * v);
      }
      logB[j] = -0.5 * (logDet + mah);
    }
    return logB;
  }

  viterbi(features) {
    const T = features.length;
    if (T === 0) return { score: -Infinity, path: [], deltaMatrix: [] };

    const delta = Array.from({ length: T }, () => new Array(this.nStates).fill(-Infinity));
    const psi = Array.from({ length: T }, () => new Array(this.nStates).fill(0));

    // t = 0
    const logB0 = this.computeLogEmission(features[0]);
    for (let j = 0; j < this.nStates; j++) {
      delta[0][j] = this.pi[j] + logB0[j];
    }

    // t = 1 ... T-1
    for (let t = 1; t < T; t++) {
      const logBt = this.computeLogEmission(features[t]);
      for (let j = 0; j < this.nStates; j++) {
        let maxVal = -Infinity;
        let maxPrev = 0;

        for (let i = 0; i < this.nStates; i++) {
          if (this.logA[i][j] > -100) {
            const score = delta[t - 1][i] + this.logA[i][j];
            if (score > maxVal) {
              maxVal = score;
              maxPrev = i;
            }
          }
        }
        delta[t][j] = maxVal + logBt[j];
        psi[t][j] = maxPrev;
      }
    }

    // Termination
    let bestLast = 0;
    let maxFinal = delta[T - 1][0];
    for (let j = 1; j < this.nStates; j++) {
      if (delta[T - 1][j] > maxFinal) {
        maxFinal = delta[T - 1][j];
        bestLast = j;
      }
    }

    // Backtrack
    const path = new Array(T);
    path[T - 1] = bestLast;
    for (let t = T - 2; t >= 0; t--) {
      path[t] = psi[t + 1][path[t + 1]];
    }

    return {
      score: maxFinal / T, // normalized per frame
      totalScore: maxFinal,
      path,
      deltaMatrix: delta
    };
  }
}

export class WebSpeechRecognizer {
  constructor(vocabulary = ["zero", "one", "two", "yes", "no", "stop"], nStates = 4) {
    this.vocabulary = vocabulary;
    this.models = {};

    vocabulary.forEach(word => {
      this.models[word] = new WebGaussianHMM(word, nStates, 13);
      // Calibrate word means dynamically for distinct acoustics
      this.calibrateWordModel(word, this.models[word]);
    });
  }

  calibrateWordModel(word, model) {
    const seed = word.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    for (let s = 0; s < model.nStates; s++) {
      for (let f = 0; f < model.nFeatures; f++) {
        model.means[s][f] = Math.sin(seed * 0.1 + s * 1.2 + f * 0.5) * 3.0;
      }
    }
  }

  recognize(features) {
    if (!features || features.length === 0) return null;

    const results = {};
    let maxScore = -Infinity;
    let predictedWord = this.vocabulary[0];

    this.vocabulary.forEach(word => {
      const res = this.models[word].viterbi(features);
      results[word] = res;
      if (res.score > maxScore) {
        maxScore = res.score;
        predictedWord = word;
      }
    });

    // Compute softmax confidence
    const scores = this.vocabulary.map(w => results[w].score);
    const maxS = Math.max(...scores);
    const exps = scores.map(s => Math.exp((s - maxS) * 0.5));
    const sumExp = exps.reduce((a, b) => a + b, 0);

    const confidences = {};
    this.vocabulary.forEach((word, idx) => {
      confidences[word] = exps[idx] / sumExp;
    });

    return {
      predictedWord,
      results,
      confidences,
      bestViterbiPath: results[predictedWord].path,
      deltaMatrix: results[predictedWord].deltaMatrix
    };
  }
}
