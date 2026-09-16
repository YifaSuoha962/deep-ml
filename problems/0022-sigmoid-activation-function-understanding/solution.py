import math

def sigmoid(z: float) -> float:
	#Your code here
	result = 1.0 / (1.0 + math.exp(-z))
	return result