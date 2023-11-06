import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def bar_plot(data, sector_names, title, xlab, ylab, labels, save_path = None, colors=None, rotation=30, fontsize=15, barWidth=0.25, dpi=300, reorder = True, gen_fig_sequence = False, order_ascending=True, contains_agg=True):
    fig = plt.subplots(dpi=dpi)

    if reorder:
        if contains_agg:
            df = pd.DataFrame(index = sector_names[:-1])
            for i, lab in enumerate(labels):
                df[lab] = data[:-1,i]
            df = df.sort_values(labels[0], ascending=order_ascending)
            sector_names[:-1] = list(df.index)
            agg_data = pd.DataFrame(data = data[-1,:].reshape((1,len(labels))), columns = labels, index=[sector_names[-1]])
            df = pd.concat([df, agg_data], axis=0)
            data = np.array(df)

        else:
            df = pd.DataFrame(index = sector_names)
            for i, lab in enumerate(labels):
                df[lab] = data[:,i]
            df = df.sort_values(labels[0], ascending=order_ascending)
            data = np.array(df)
            sector_names = list(df.index)

    if gen_fig_sequence:
        for fs in range(len(labels)):
            data[:, :fs] = 0

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

            plt.xlabel(xlab, fontweight='bold', fontsize=fontsize)
            plt.ylabel(ylab, fontweight='bold', fontsize=fontsize)

            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path + str(fs) + '.png', dpi=dpi, transparent=True)
            plt.close()
    else:
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

            plt.xlabel(xlab, fontweight='bold', fontsize=fontsize)
            plt.ylabel(ylab, fontweight='bold', fontsize=fontsize)

            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path + '.png', dpi=dpi, transparent=True)