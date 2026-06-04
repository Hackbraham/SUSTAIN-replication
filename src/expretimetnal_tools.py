"""
This file keeps a few general, sustain-related functions that are employed to run experiments.
"""


from src.sustain import SUSTAIN
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def run_sustain_on_data(model: SUSTAIN, data: pd.DataFrame, queried_dim: int):
    """
    Runs a single iteration of SUSTAIN experiment on a single dataset.
    """

    # keep track of stats
    index = []
    response = []
    probability = []
    correct = []
    n_clusters = []
    winner = []
    recruited = []

    for i, stimulus in enumerate(data):
        res = model.present_stimulus(stimulus, queried_dim=queried_dim)

        index.append(i)
        response.append(res['response'])
        probability.append(res['prob'])
        correct.append(res['correct'])
        n_clusters.append(res['n_clusters'])
        winner.append(res['winner'])
        recruited.append(res['recruited'])

    results = pd.DataFrame({
        "index": index,
        "response": response,
        "probability": probability,
        "correct": correct,
        "n_clusters": n_clusters,
        "winner": winner,
        "recruited": recruited
    })

    results.set_index("index", inplace=True)
    return results, model


def encode_stimuli(data: pd.DataFrame):
    """
    A simple helper that encodes the stimuli for SUSTAIN.
    """
    # convert to a list
    stimuli = []
    for row in data.itertuples(index=False):
        row_list = list(row)
        stimuli.append(row_list)

    # the list consts of string, we need numbers
    encoder = OrdinalEncoder()
    encoded_stimuli = encoder.fit_transform(stimuli).astype(int).tolist()

    # for debugging
    # for i, cats in enumerate(encoder.categories_):
    #     print(f"Dimension {i}: {list(cats)}")

    # compute the size of dimensions
    dim_sizes = [len(categories) for categories in encoder.categories_]
    # print(f"dim_sizes: {dim_sizes}")

    return encoded_stimuli, dim_sizes