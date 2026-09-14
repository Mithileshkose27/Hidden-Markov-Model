import numpy as np
from .mfcc_extractor import MFCCExtractor

class SyntheticSpeechGenerator:
    """
    Generates synthetic speech audio waveforms and MFCC feature sequences
    for isolated word speech recognition testing (e.g., "zero", "one", "two", "yes", "no", "stop").
    """

    # Formant frequencies (F1, F2, F3 in Hz) for key vowels and consonants
    SPEECH_PHONEMES = {
        'z': {'f1': 250,  'f2': 1800, 'f3': 2600, 'noise': 0.3},
        'e': {'f1': 530,  'f2': 1840, 'f3': 2480, 'noise': 0.05},
        'r': {'f1': 350,  'f2': 1300, 'f3': 1600, 'noise': 0.05},
        'o': {'f1': 500,  'f2': 1000, 'f3': 2300, 'noise': 0.05},
        'w': {'f1': 300,  'f2': 800,  'f3': 2200, 'noise': 0.05},
        'ah':{'f1': 700,  'f2': 1100, 'f3': 2400, 'noise': 0.05},
        'n': {'f1': 250,  'f2': 1400, 'f3': 2200, 'noise': 0.1},
        't': {'f1': 200,  'f2': 1500, 'f3': 2500, 'noise': 0.5},
        'u': {'f1': 320,  'f2': 800,  'f3': 2300, 'noise': 0.05},
        'y': {'f1': 280,  'f2': 2250, 'f3': 2890, 'noise': 0.05},
        'eh':{'f1': 550,  'f2': 1800, 'f3': 2500, 'noise': 0.05},
        's': {'f1': 400,  'f2': 2000, 'f3': 3000, 'noise': 0.6},
        'p': {'f1': 150,  'f2': 1100, 'f3': 2200, 'noise': 0.4},
        'sil': {'f1': 0,  'f2': 0,    'f3': 0,    'noise': 0.01}
    }

    WORD_PHONEME_MAP = {
        "zero": ["z", "e", "r", "o"],
        "one":  ["w", "ah", "n"],
        "two":  ["t", "u"],
        "yes":  ["y", "eh", "s"],
        "no":   ["n", "o"],
        "stop": ["s", "t", "o", "p"]
    }

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.extractor = MFCCExtractor(sample_rate=sample_rate)

    def generate_phoneme_sound(self, phoneme_key, duration=0.15, pitch=120.0):
        """Synthesizes formant-filtered acoustic signal for a phoneme."""
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        p_info = self.SPEECH_PHONEMES.get(phoneme_key, self.SPEECH_PHONEMES['sil'])

        if phoneme_key == 'sil':
            return np.random.normal(0, p_info['noise'], len(t))

        # Fundamental pitch & harmonics
        signal = np.zeros_like(t)
        for h in range(1, 5):
            signal += (1.0 / h) * np.sin(2 * np.pi * (pitch * h) * t)

        # Formant resonances (F1, F2, F3 filter simulation)
        f1, f2, f3 = p_info['f1'], p_info['f2'], p_info['f3']
        if f1 > 0:
            signal += 0.8 * np.sin(2 * np.pi * f1 * t)
        if f2 > 0:
            signal += 0.5 * np.sin(2 * np.pi * f2 * t)
        if f3 > 0:
            signal += 0.3 * np.sin(2 * np.pi * f3 * t)

        # Add fricative/consonant noise envelope
        noise = np.random.normal(0, p_info['noise'], len(t))
        signal = signal * (1.0 - p_info['noise']) + noise

        # Apply smooth attack & decay amplitude envelope
        envelope = np.sin(np.pi * np.linspace(0, 1, len(t)))
        signal *= envelope
        return signal

    def generate_word_audio(self, word, duration_per_phoneme=0.12, noise_level=0.02):
        """Synthesizes raw audio waveform for a spoken word."""
        phonemes = self.WORD_PHONEME_MAP.get(word.lower(), ["sil", "o", "sil"])
        audio_segments = [self.generate_phoneme_sound("sil", duration=0.05)]
        
        for ph in phonemes:
            # Vary phoneme duration slightly
            dur = duration_per_phoneme * np.random.uniform(0.85, 1.15)
            pitch = np.random.uniform(110.0, 130.0)
            seg = self.generate_phoneme_sound(ph, duration=dur, pitch=pitch)
            audio_segments.append(seg)

        audio_segments.append(self.generate_phoneme_sound("sil", duration=0.05))
        full_audio = np.concatenate(audio_segments)
        
        # Add background white noise
        if noise_level > 0:
            full_audio += np.random.normal(0, noise_level, len(full_audio))
            
        # Peak normalization
        max_val = np.max(np.abs(full_audio))
        if max_val > 0:
            full_audio = full_audio / max_val
            
        return full_audio

    def generate_dataset(self, words=("zero", "one", "two", "yes", "no", "stop"),
                         samples_per_word=10, include_deltas=True):
        """
        Generates training and testing dataset of word MFCC feature sequences.
        Returns dict: { word: [ feature_matrix_1, feature_matrix_2, ... ] }
        """
        dataset = {}
        for w in words:
            dataset[w] = []
            for _ in range(samples_per_word):
                audio = self.generate_word_audio(w)
                features = self.extractor.extract_features(audio, include_deltas=include_deltas)
                dataset[w].append(features)
        return dataset
