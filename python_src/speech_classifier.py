import numpy as np
from .hmm_core import GaussianHMM
from .mfcc_extractor import MFCCExtractor

class HMMSpeechRecognizer:
    """
    Isolated Word Speech Recognition Classifier using Hidden Markov Models.
    Maintains a dictionary of trained word HMMs (one per vocabulary word).
    """

    def __init__(self, vocabulary=("zero", "one", "two", "yes", "no", "stop"),
                 n_states=4, n_features=39, topology="left_right"):
        self.vocabulary = list(vocabulary)
        self.n_states = n_states
        self.n_features = n_features
        self.topology = topology
        self.extractor = MFCCExtractor()
        
        # Instantiate an HMM model for each word in vocabulary
        self.models = {
            word: GaussianHMM(n_states=n_states, n_features=n_features, topology=topology)
            for word in self.vocabulary
        }
        self.is_trained = False

    def train(self, dataset, n_iter=12):
        """
        Trains each word HMM using Baum-Welch (EM) algorithm on dataset.
        dataset dict: { word: [ feature_matrix_1, feature_matrix_2, ... ] }
        """
        print("Training HMM Speech Recognition Models...")
        for word in self.vocabulary:
            if word in dataset and len(dataset[word]) > 0:
                print(f"  --> Fitting HMM for word '{word}' ({len(dataset[word])} sequences)...")
                self.models[word].fit(dataset[word], n_iter=n_iter)
            else:
                print(f"  [Warning] No training data found for word '{word}'")
        self.is_trained = True
        print("Model training complete!")

    def predict_features(self, X):
        """
        Predicts word identity for a given MFCC feature matrix X (T, D).
        Returns dict with:
        - predicted_word (str)
        - scores (dict of word -> log_likelihood_score)
        - confidences (dict of word -> normalized probability)
        - best_viterbi_path (list of states for predicted word)
        """
        scores = {}
        viterbi_paths = {}

        for word, model in self.models.items():
            best_path, v_score, _ = model.viterbi(X)
            # Normalized score per frame
            norm_score = v_score / float(len(X))
            scores[word] = norm_score
            viterbi_paths[word] = best_path

        # Find best candidate word
        best_word = max(scores, key=scores.get)

        # Compute softmax probabilities over log scores for confidence display
        score_vals = np.array([scores[w] for w in self.vocabulary])
        max_s = np.max(score_vals)
        exp_s = np.exp(score_vals - max_s)
        probs = exp_s / np.sum(exp_s)
        
        confidences = {self.vocabulary[i]: float(probs[i]) for i in range(len(self.vocabulary))}

        return {
            "predicted_word": best_word,
            "scores": scores,
            "confidences": confidences,
            "viterbi_path": viterbi_paths[best_word],
            "all_viterbi_paths": viterbi_paths
        }

    def predict_audio(self, audio_signal, include_deltas=True):
        """Extracts features from raw audio signal and predicts word."""
        features = self.extractor.extract_features(audio_signal, include_deltas=include_deltas)
        return self.predict_features(features)

    def evaluate(self, test_dataset):
        """
        Evaluates classifier on test dataset.
        Returns accuracy and confusion matrix dict.
        """
        correct = 0
        total = 0
        conf_matrix = {w1: {w2: 0 for w2 in self.vocabulary} for w1 in self.vocabulary}

        for true_word, sequences in test_dataset.items():
            if true_word not in self.vocabulary:
                continue
            for seq in sequences:
                res = self.predict_features(seq)
                pred_word = res["predicted_word"]
                conf_matrix[true_word][pred_word] += 1
                if pred_word == true_word:
                    correct += 1
                total += 1

        accuracy = correct / float(total) if total > 0 else 0.0
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "confusion_matrix": conf_matrix
        }
