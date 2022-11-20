import numpy as np

h_ids = np.load("hawkes_transformer_correct_ids.npy")
s_ids = np.load("sole_transformer_correct_ids.npy")

print(len(h_ids), len(s_ids))

h_more_ids = []
for i in range(len(h_ids)):
    if h_ids[i] not in s_ids:

        h_more_ids.append(h_ids[i])

print(len(h_more_ids))