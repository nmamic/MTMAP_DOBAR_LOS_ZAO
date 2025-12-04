import numpy as np
import soundfile
from scipy.signal import ShortTimeFFT, get_window
from nmf import *
import pickle

N = 1000
k = 7

sound, sr = soundfile.read("test_piano_1.wav")
sound = np.mean(sound, axis = 1)

SFT_sound = ShortTimeFFT(get_window("hann", 1024), hop = 512, fs = sr)
Z = SFT_sound.stft(sound)
X = np.abs(Z)**2 + 1e-12

NMF_sound = NMF(br_iter = 1000, br_komp=k)

results = []
for i in range(N):
    print(f"Run: {i+1}/1000")
    W = np.random.rand(X.shape[0], k) + np.ones((X.shape[0], k))
    H = np.random.rand(k, X.shape[1]) + np.ones((k, X.shape[1]))
    result = NMF_sound.ard_is_nmf(X, 1e-3, 250, W = W, H = H, pruning_threshold=10, suppress_print=True)
    results.append(result["mask"])

with open('results_three_notes_compact.pkl', 'wb') as f:
    pickle.dump(results, f)

f.close
