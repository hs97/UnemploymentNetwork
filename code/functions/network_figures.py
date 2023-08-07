import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def bar_plot(data, sector_names, title, xlab, ylab, labels, save_path, colors=None, rotation=30, fontsize=15, barWidth=0.25, dpi=300):
    #creating arrays
    br = np.zeros((len(sector_names), data.shape[1]))

    #initializing
    br[:, 0] = np.arange(len(sector_names))
    if colors == None:
        plt.bar(br[:,0], data[:,0], width=barWidth, label=labels[0]) 
    else:  
        plt.bar(br[:,0], data[:,0], width=barWidth,
               color=colors[0], label=labels[0])
    
    #looping through other networks
    for i in range(1, data.shape[1]):
        br[:, i] = [x + barWidth for x in br[:, i-1]]
        if colors == None:
            plt.bar(br[:, i], data[:, i], width=barWidth, label=labels[i])
        else:
            plt.bar(br[:, i], data[:, i], width=barWidth,
                    color=colors[i], label=labels[i])
    plt.xticks([r + barWidth for r in range(len(sector_names))],
                sector_names, rotation=rotation, fontsize=6)

    plt.title(title, fontweight='bold', fontsize=fontsize)
    plt.xlabel(xlab, fontweight='bold', fontsize=fontsize)
    plt.ylabel(ylab, fontweight='bold', fontsize=fontsize)

    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, dpi=dpi, transparent=True)  