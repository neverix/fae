import jax.numpy as np

# see data_analyzer.py

# http://neilsloane.com/hadamard/
had_12 = """+-----------
++-+---+++-+
+++-+---+++-
+-++-+---+++
++-++-+---++
+++-++-+---+
++++-++-+---
+-+++-++-+--
+--+++-++-+-
+---+++-++-+
++---+++-++-
+-+---+++-++"""
had_12 = np.array([[1 if c == "+" else -1 for c in row] for row in had_12.split("\n") if row])
def hadamard_matrix(n):
    if n == 1:
        return np.array([[1]])
    elif n == 12:
        return had_12
    else:
        h = hadamard_matrix(n // 2)
        return np.block([[h, h], [h, -h]])