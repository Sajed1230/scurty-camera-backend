import time
import numpy as np

for n in (100, 1_000, 10_000, 100_000):
    vectors = np.random.randn(n, 512).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    probe = vectors[0]

    start = time.perf_counter()
    for _ in range(100):
        scores = vectors @ probe
        best = int(np.argmax(scores))
    ms = (time.perf_counter() - start) / 100 * 1000
    print(f"{n:>7} signatures: {ms:.2f} ms per match")