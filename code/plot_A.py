import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    A = pd.read_csv("data/clean/A.csv")
    A = A.set_index('BEA_sector')
    sns.heatmap(A, cmap="crest")
    plt.show()
