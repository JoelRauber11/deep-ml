import numpy as np

def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	arr = np.array(a)

	if arr.shape[1] != len(b):
		return -1

	return np.dot(arr, b)