import numpy as np

class MFCCExtractor:
    """
    Speech Signal Processing & Mel-Frequency Cepstral Coefficients (MFCC) Extractor.
    Extracts static MFCCs, energy, and dynamic derivative features (Delta & Delta-Delta).
    """

    def __init__(self, sample_rate=16000, frame_size=0.025, frame_stride=0.010,
                 num_cepstra=13, num_filters=26, nfft=512, pre_emphasis=0.97):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.frame_stride = frame_stride
        self.num_cepstra = num_cepstra
        self.num_filters = num_filters
        self.nfft = nfft
        self.pre_emphasis = pre_emphasis

    def pre_emphasis_filter(self, signal):
        """Applies pre-emphasis filter to boost high frequencies."""
        return np.append(signal[0], signal[1:] - self.pre_emphasis * signal[:-1])

    def frame_signal(self, signal):
        """Splits signal into overlapping temporal frames."""
        frame_length = int(round(self.frame_size * self.sample_rate))
        frame_step = int(round(self.frame_stride * self.sample_rate))
        signal_length = len(signal)
        
        if signal_length <= frame_length:
            num_frames = 1
        else:
            num_frames = 1 + int(np.ceil((signal_length - frame_length) / float(frame_step)))

        pad_signal_length = (num_frames - 1) * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(signal, z)

        indices = (
            np.tile(np.arange(0, frame_length), (num_frames, 1)) +
            np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        )
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        
        # Apply Hamming window
        frames *= np.hamming(frame_length)
        return frames

    def get_mel_filterbank(self):
        """Constructs Mel-frequency triangular filterbank matrix."""
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (self.sample_rate / 2.0) / 700.0))
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.num_filters + 2)
        hz_points = (700 * (10**(mel_points / 2595.0) - 1))
        bin_points = np.floor((self.nfft + 1) * hz_points / self.sample_rate).astype(int)

        fbank = np.zeros((self.num_filters, int(np.floor(self.nfft / 2 + 1))))
        for m in range(1, self.num_filters + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            for k in range(f_m_minus, f_m):
                if f_m != f_m_minus:
                    fbank[m - 1, k] = (k - bin_points[m - 1]) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus != f_m:
                    fbank[m - 1, k] = (bin_points[m + 1] - k) / (f_m_plus - f_m)
                    
        return fbank

    def compute_mfcc(self, signal):
        """
        Extracts 13 MFCC features per frame from raw 1D audio signal.
        Returns array of shape (N_frames, 13).
        """
        if len(signal) == 0:
            return np.zeros((0, self.num_cepstra))

        # 1. Pre-emphasis
        emphasized_signal = self.pre_emphasis_filter(signal)
        
        # 2. Framing & Windowing
        frames = self.frame_signal(emphasized_signal)
        
        # 3. Fourier Transform & Power Spectrum
        mag_frames = np.abs(np.fft.rfft(frames, self.nfft))
        pow_frames = ((1.0 / self.nfft) * (mag_frames ** 2))
        
        # Avoid zero log issue
        pow_frames = np.where(pow_frames == 0, np.finfo(float).eps, pow_frames)
        
        # 4. Mel Filterbank Integration
        fbank = self.get_mel_filterbank()
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks) # Log mel energy
        
        # 5. Discrete Cosine Transform (DCT)
        num_frames = filter_banks.shape[0]
        n_filters = filter_banks.shape[1]
        dct_matrix = np.zeros((self.num_cepstra, n_filters))
        for n in range(self.num_cepstra):
            dct_matrix[n, :] = np.cos(np.pi * n * (2 * np.arange(n_filters) + 1) / (2.0 * n_filters))
        
        mfcc = np.dot(filter_banks, dct_matrix.T)
        
        # Sinusoidal Lifter to balance high and low cepstral components
        ncep = mfcc.shape[1]
        n = np.arange(ncep)
        cep_lifter = 22
        lift = 1 + (cep_lifter / 2.0) * np.sin(np.pi * n / cep_lifter)
        mfcc *= lift
        
        # Mean normalization for channel noise robustness
        mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
        return mfcc

    def compute_deltas(self, features, N=2):
        """Computes delta (first derivative) or delta-delta coefficients."""
        if len(features) == 0:
            return np.zeros_like(features)
        
        num_frames = len(features)
        deltas = np.zeros_like(features)
        padded = np.pad(features, ((N, N), (0, 0)), mode='edge')
        
        denominator = 2 * sum([i**2 for i in range(1, N + 1)])
        for t in range(num_frames):
            delta_sum = np.zeros(features.shape[1])
            for n in range(1, N + 1):
                delta_sum += n * (padded[t + N + n] - padded[t + N - n])
            deltas[t] = delta_sum / denominator
        return deltas

    def extract_features(self, signal, include_deltas=True):
        """
        Extracts full feature matrix for audio signal.
        Static (13) + Delta (13) + Delta-Delta (13) = 39 features per frame if include_deltas=True.
        """
        mfcc = self.compute_mfcc(signal)
        if not include_deltas:
            return mfcc
            
        d_mfcc = self.compute_deltas(mfcc)
        dd_mfcc = self.compute_deltas(d_mfcc)
        return np.hstack((mfcc, d_mfcc, dd_mfcc))
