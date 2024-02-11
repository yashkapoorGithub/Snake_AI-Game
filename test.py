import numpy as np
softmax_outputs = np.array([[0.7, 0.1, 0.2],
[0.1, 0.5, 0.4],
[0.02, 0.9, 0.08]])
class_targets = np.array([[1, 0, 0],
[0, 1, 0],
[0, 1, 0]])
# Probabilities for target values -
# only if categorical labels
#if len(class_targets.shape) == 1:
#print(len(class_targets.shape))
correct_confidences = softmax_outputs[
range(len(softmax_outputs)),
class_targets
]
#print(correct_confidences)
