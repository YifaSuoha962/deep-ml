import numpy as np

def phi_transform(data: list[float], degree: int) -> list[list[float]]:
	"""
	Perform a Phi Transformation to map input features into a higher-dimensional space by generating polynomial features.

	Args:
		data (list[float]): A list of numerical values to transform.
		degree (int): The degree of the polynomial expansion.

	"""
	# Your code here
	if degree < 0:
		return []
	else:
		phis_list = []
		for x in data:
			phis = [x ** i for i in range(degree + 1)]
			phis_list.append(phis)
	return phis_list