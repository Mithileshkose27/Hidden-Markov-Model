import os
import sys
import numpy as np

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from python_src.synthetic_speech_generator import SyntheticSpeechGenerator
from python_src.speech_classifier import HMMSpeechRecognizer
from python_src.hmm_core import GaussianHMM

def run_demo():
    print("=" * 70)
    print("      HIDDEN MARKOV MODEL (HMM) FOR SPEECH RECOGNITION DEMO      ")
    print("=" * 70)

    vocab = ["zero", "one", "two", "yes", "no", "stop"]
    print(f"\n1. Target Vocabulary ({len(vocab)} words): {vocab}")
    
    # 1. Generate Synthetic Speech Datasets
    print("\n2. Generating synthetic formant speech audio & MFCC feature samples...")
    gen = SyntheticSpeechGenerator(sample_rate=16000)
    
    train_dataset = gen.generate_dataset(words=vocab, samples_per_word=12, include_deltas=True)
    test_dataset  = gen.generate_dataset(words=vocab, samples_per_word=5, include_deltas=True)
    
    sample_feat = train_dataset["zero"][0]
    print(f"   -> Sample MFCC feature matrix shape: {sample_feat.shape} (Frames x Features)")

    # 2. Train HMM Recognizer
    print("\n3. Building & Training Left-to-Right Bakis Gaussian HMMs...")
    recognizer = HMMSpeechRecognizer(vocabulary=vocab, n_states=4, n_features=39, topology="left_right")
    recognizer.train(train_dataset, n_iter=10)

    # 3. Evaluate Classifier
    print("\n4. Evaluating Model on Test Dataset...")
    eval_results = recognizer.evaluate(test_dataset)
    acc = eval_results["accuracy"] * 100.0
    print(f"   -> Overall Test Accuracy: {acc:.2f}% ({eval_results['correct']}/{eval_results['total']} samples)")

    # Print Confusion Matrix
    print("\n   Confusion Matrix (Rows: True, Cols: Predicted):")
    header = f"{'True\\Pred':<10}" + "".join([f"{w:>8}" for w in vocab])
    print("   " + header)
    print("   " + "-" * len(header))
    for tw in vocab:
        row_str = f"{tw:<10}" + "".join([f"{eval_results['confusion_matrix'][tw][pw]:>8}" for pw in vocab])
        print("   " + row_str)

    # 4. Demonstrate Viterbi Decoding & State Alignment on a Test Audio Sample
    print("\n5. Detailed Viterbi State Alignment Demo on spoken word 'yes':")
    test_audio = gen.generate_word_audio("yes")
    res = recognizer.predict_audio(test_audio, include_deltas=True)

    print(f"   -> Predicted Word: '{res['predicted_word']}'")
    print(f"   -> Confidence Scores:")
    for w in vocab:
        bar = "#" * int(res['confidences'][w] * 20)
        print(f"      {w:<6}: {res['confidences'][w]*100:6.2f}% | {bar}")

    path = res['viterbi_path']
    print(f"\n   -> Viterbi State Alignment Path over {len(path)} audio frames:")
    print(f"      Frame-by-frame state sequence: {path}")

    # Count frames per HMM state phase
    state_counts = {}
    for st in path:
        state_counts[st] = state_counts.get(st, 0) + 1
    print("      State duration breakdown:")
    for st in sorted(state_counts.keys()):
        print(f"        State {st} (Phase {st+1}): {state_counts[st]} frames ({state_counts[st]*10} ms)")

    print("\n=" * 70)
    print("HMM Speech Recognition Demo completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    run_demo()
