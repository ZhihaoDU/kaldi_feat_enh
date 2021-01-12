import numpy as np


def basic_collection_function(sample_list):
    batch_dict = {}
    for key, value in sample_list[0].items():
        batch_dict[key] = []

    for one_sample in sample_list:
        for key, value in one_sample.items():
            batch_dict[key].append(value)

    batch = {}
    for key, data in batch_dict.items():
        batch[key] = np.array(data)

    assert "speech" in batch.keys() and "noisy" in batch.keys() and "length" in batch.keys(), \
        "Speech, noisy and length must be in a mini-batch"
    return batch
