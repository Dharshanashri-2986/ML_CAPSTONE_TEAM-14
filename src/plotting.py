"""Shared plotting style (colourblind-friendly, per guideline 7.3)."""
import matplotlib.pyplot as plt
import seaborn as sns


def set_style():
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.rcParams.update({
        "figure.dpi": 100,
        "axes.titleweight": "bold",
        "savefig.bbox": "tight",
    })
