import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from python_src.mfcc_extractor import MFCCExtractor
from python_src.hmm_core import GaussianHMM, log_sum_exp
from python_src.synthetic_speech_generator import SyntheticSpeechGenerator
from python_src.speech_classifier import HMMSpeechRecognizer


class TestHMMSpeechRecognition(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 16000
        self.extractor = MFCCExtractor(sample_rate=self.sample_rate)
        self.gen = SyntheticSpeechGenerator(sample_rate=self.sample_rate)

    def test_log_sum_exp(self):
        log_vals = np.array([-2.0, -1.0, -3.0])
        val = log_sum_exp(log_vals)
        expected = np.log(np.sum(np.exp(log_vals)))
        self.assertAlmostEqual(val, expected, places=5)

    def test_mfcc_extractor(self):
        audio = self.gen.generate_word_audio("zero")
        mfcc = self.extractor.compute_mfcc(audio)
        self.assertGreater(mfcc.shape[0], 0)
        self.assertEqual(mfcc.shape[1], 13)

        full_feat = self.extractor.extract_features(audio, include_deltas=True)
        self.assertEqual(full_feat.shape[1], 39)

    def test_hmm_bakis_topology(self):
        hmm = GaussianHMM(n_states=4, n_features=39, topology="left_right")
        # Check start state
        self.assertEqual(hmm.pi[0], 1.0)
        self.assertEqual(hmm.pi[1], 0.0)

        # Check Left-to-Right constraints (no backward transitions)
        for i in range(4):
            for j in range(i):
                self.assertEqual(hmm.A[i, j], 0.0)

        # Row stochasticity
        row_sums = np.sum(hmm.A, axis=1)
        np.testing.assert_allclose(row_sums, 1.0)

    def test_forward_backward_consistency(self):
        audio = self.gen.generate_word_audio("one")
        X = self.extractor.extract_features(audio)
        hmm = GaussianHMM(n_states=3, n_features=39)
        
        # Initialize means
        hmm.fit([X], n_iter=1)
        
        log_alpha, p_forward = hmm.forward(X)
        log_beta = hmm.backward(X)

        # Check forward log likelihood is finite
        self.assertTrue(np.isfinite(p_forward))

        # Check log_alpha + log_beta sum at any frame t equals total log likelihood p_forward
        T = len(X)
        for t in range(T):
            frame_p = log_sum_exp(log_alpha[t, :] + log_beta[t, :])
            self.assertAlmostEqual(frame_p, p_forward, places=3)

    def test_viterbi_decoding(self):
        audio = self.gen.generate_word_audio("two")
        X = self.extractor.extract_features(audio)
        hmm = GaussianHMM(n_states=4, n_features=39)
        hmm.fit([X], n_iter=2)

        path, v_score, delta = hmm.viterbi(X)
        self.assertEqual(len(path), len(X))
        # Path must start at state 0 and end at valid state <= 3
        self.assertEqual(path[0], 0)
        self.assertGreaterEqual(path[-1], 0)
        self.assertLess(path[-1], 4)

        # Non-decreasing state sequence for Left-to-Right Bakis topology
        for i in range(1, len(path)):
            self.assertGreaterEqual(path[i], path[i - 1])

    def test_baum_welch_training(self):
        sequences = [self.extractor.extract_features(self.gen.generate_word_audio("stop")) for _ in range(5)]
        hmm = GaussianHMM(n_states=3, n_features=39)
        
        # Initial score
        hmm.fit(sequences, n_iter=1)
        s1 = sum([hmm.score(s) for s in sequences])

        # More iterations should improve score
        hmm.fit(sequences, n_iter=6)
        s2 = sum([hmm.score(s) for s in sequences])
        self.assertGreaterEqual(s2, s1 - 1e-3)

    def test_speech_classifier_pipeline(self):
        vocab = ["zero", "one", "two"]
        train_data = self.gen.generate_dataset(words=vocab, samples_per_word=6)
        test_data  = self.gen.generate_dataset(words=vocab, samples_per_word=3)

        recognizer = HMMSpeechRecognizer(vocabulary=vocab, n_states=3, n_features=39)
        recognizer.train(train_data, n_iter=8)

        eval_res = recognizer.evaluate(test_data)
        self.assertGreaterEqual(eval_res["accuracy"], 0.70)


if __name__ == "__main__":
    unittest.main()
