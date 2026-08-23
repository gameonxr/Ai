import numpy as np


def normalize_quaternion(quaternion):
    q = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(q)
    return (q / norm if norm else np.array([0.0, 0.0, 0.0, 1.0])).tolist()


def finite_vector(values, size=3):
    vector = np.asarray(values, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"Expected {size} finite values")
    return vector
