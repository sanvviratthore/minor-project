"""Feature definition shared by training, explanation and optimisation."""
import numpy as np
import pandas as pd

FEATURES = ["m", "z1", "z2", "b", "x1", "x2", "logT1", "logn1", "Ra", "logeta",
            "oil_pao", "oil_pag"]


def featurise(d):
    X = pd.DataFrame({
        "m": d["m"], "z1": d["z1"], "z2": d["z2"], "b": d["b"], "x1": d["x1"], "x2": d["x2"],
        "logT1": np.log(d["T1"]), "logn1": np.log(d["n1"]), "Ra": d["Ra"], "logeta": np.log(d["eta"]),
        "oil_pao": (d["oil_type"] == "pao").astype(float),
        "oil_pag": (d["oil_type"] == "pag").astype(float)})
    return X[FEATURES]
