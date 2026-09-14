/**
 * Web Audio Signal Processing & Synthetic Speech Audio Generator
 */

// Formant definitions for client-side synthetic speech synthesis
export const PRESET_WORDS = ["zero", "one", "two", "yes", "no", "stop"];

const PHONEME_DATA = {
  z:  { f1: 250, f2: 1800, f3: 2600, noise: 0.35 },
  e:  { f1: 530, f2: 1840, f3: 2480, noise: 0.05 },
  r:  { f1: 350, f2: 1300, f3: 1600, noise: 0.05 },
  o:  { f1: 500, f2: 1000, f3: 2300, noise: 0.05 },
  w:  { f1: 300, f2: 800,  f3: 2200, noise: 0.05 },
  ah: { f1: 700, f2: 1100, f3: 2400, noise: 0.05 },
  n:  { f1: 250, f2: 1400, f3: 2200, noise: 0.10 },
  t:  { f1: 200, f2: 1500, f3: 2500, noise: 0.50 },
  u:  { f1: 320, f2: 800,  f3: 2300, noise: 0.05 },
  y:  { f1: 280, f2: 2250, f3: 2890, noise: 0.05 },
  eh: { f1: 550, f2: 1800, f3: 2500, noise: 0.05 },
  s:  { f1: 400, f2: 2000, f3: 3000, noise: 0.65 },
  p:  { f1: 150, f2: 1100, f3: 2200, noise: 0.40 },
  sil:{ f1: 0,   f2: 0,    f3: 0,    noise: 0.01 }
};

const WORD_MAP = {
  zero: ["z", "e", "r", "o"],
  one:  ["w", "ah", "n"],
  two:  ["t", "u"],
  yes:  ["y", "eh", "s"],
  no:   ["n", "o"],
  stop: ["s", "t", "o", "p"]
};

export function generateSyntheticWordAudio(word, sampleRate = 16000) {
  const phonemes = WORD_MAP[word.toLowerCase()] || ["sil", "o", "sil"];
  const list = ["sil", ...phonemes, "sil"];
  const segments = [];

  list.forEach(ph => {
    const dur = ph === "sil" ? 0.05 : 0.12;
    const numSamples = Math.floor(sampleRate * dur);
    const p = PHONEME_DATA[ph] || PHONEME_DATA["sil"];
    const buffer = new Float32Array(numSamples);

    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      let val = 0;
      if (ph !== "sil") {
        for (let h = 1; h <= 4; h++) {
          val += (1.0 / h) * Math.sin(2 * Math.PI * 120 * h * t);
        }
        if (p.f1 > 0) val += 0.8 * Math.sin(2 * Math.PI * p.f1 * t);
        if (p.f2 > 0) val += 0.5 * Math.sin(2 * Math.PI * p.f2 * t);
        if (p.f3 > 0) val += 0.3 * Math.sin(2 * Math.PI * p.f3 * t);
      }
      const noise = (Math.random() * 2 - 1) * p.noise;
      val = val * (1 - p.noise) + noise;

      // Envelope
      const env = Math.sin(Math.PI * (i / numSamples));
      buffer[i] = val * env;
    }
    segments.push(buffer);
  });

  // Concatenate
  const totalLen = segments.reduce((sum, seg) => sum + seg.length, 0);
  const result = new Float32Array(totalLen);
  let offset = 0;
  segments.forEach(seg => {
    result.set(seg, offset);
    offset += seg.length;
  });

  // Normalize
  let maxAbs = 0;
  for (let i = 0; i < totalLen; i++) {
    if (Math.abs(result[i]) > maxAbs) maxAbs = Math.abs(result[i]);
  }
  if (maxAbs > 0) {
    for (let i = 0; i < totalLen; i++) result[i] /= maxAbs;
  }

  return result;
}

export function extractWebMFCC(audioSignal, sampleRate = 16000, numCepstra = 13) {
  const frameLength = Math.floor(0.025 * sampleRate);
  const frameStep = Math.floor(0.010 * sampleRate);
  const signalLen = audioSignal.length;

  if (signalLen < frameLength) return [];

  const numFrames = 1 + Math.floor((signalLen - frameLength) / frameStep);
  const mfccMatrix = [];

  for (let f = 0; f < numFrames; f++) {
    const start = f * frameStep;
    const frame = new Float32Array(frameLength);

    for (let i = 0; i < frameLength; i++) {
      // Pre-emphasis + Hamming
      const s = audioSignal[start + i] - 0.97 * (start + i > 0 ? audioSignal[start + i - 1] : 0);
      const window = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (frameLength - 1));
      frame[i] = s * window;
    }

    // 13 Synthetic energy/cepstral bands for web real-time display
    const cepstra = new Float32Array(numCepstra);
    let totalEnergy = 0;
    for (let i = 0; i < frameLength; i++) totalEnergy += frame[i] * frame[i];

    cepstra[0] = Math.log(Math.max(totalEnergy, 1e-5));
    for (let c = 1; c < numCepstra; c++) {
      let bandEnergy = 0;
      const k = c * 3;
      for (let i = 0; i < frameLength - k; i += 2) {
        bandEnergy += Math.abs(frame[i] * frame[i + k]);
      }
      cepstra[c] = Math.cos(c * 0.4) * Math.log(Math.max(bandEnergy, 1e-4));
    }
    mfccMatrix.push(Array.from(cepstra));
  }

  return mfccMatrix;
}
