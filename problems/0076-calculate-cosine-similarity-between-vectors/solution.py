import numpy as np

def cosine_similarity(v1, v2):
	"""
	Calculate the cosine_similarity of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The cosine_similarity of the two vectors.
	"""
	if v1.shape != v2.shape:
		raise ValueError("Both input vectors must have the same shape.")

	if v1.size == 0 or v2.size == 0:
		raise ValueError("Vectors cannot be empty")

	n1 = np.linalg.norm(v1)
	n2 = np.linalg.norm(v2)

	if n1 == 0 or n2 == 0:
		raise ValueError("Input vectors cannot be empty or have zero magnitude")

	return float(np.dot(v1, v2) / (n1 * n2))