import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	if len(gradient) > 0:
		gradient = np.array(gradient)
		
		if np.all(gradient == 0):
			return {
				'magnitude': 0,
				'direction': np.zeros(gradient.shape),
				'descent_direction': np.zeros(gradient.shape)
			}	

		mag = np.sqrt(np.array([x ** 2 for x in gradient]).sum())

		direction = [x / mag for x in gradient]
		d_dir = [-(x / mag) for x in gradient]
	else:
		raise ValueError("List is empty")

	return {
		'magnitude': mag,
		'direction': direction,
		'descent_direction': d_dir
	}