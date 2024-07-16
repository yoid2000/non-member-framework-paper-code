import pandas as pd
import json
import pprint
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":
    with open('resultsRecall.json', 'r') as f:
        res = json.load(f)

    df = pd.DataFrame(res)
    print(df.head())

    doLegend = True
    if doLegend:
        fig = plt.figure(figsize=(3.5, 2.0))
        ax = fig.add_subplot(111)  # 111 means 1 row, 1 column, first plot
        fileNamePng = 'prec-recall.png'
        fileNamePdf = 'prec-recall.pdf'
    else:
        fig = plt.figure(figsize=(4, 2.5))
        ax = fig.add_subplot(111)  # 111 means 1 row, 1 column, first plot
        fileNamePng = 'prec-recall-nolegend.png'
        fileNamePdf = 'prec-recall-nolegend.pdf'
    scatter = sns.scatterplot(data=df, x='recall', y='prec', hue='target', style='target', legend=doLegend)
    if doLegend:
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
        plt.legend(fontsize=6, ncol=2)  # using a named size
    #plt.xscale('log')
    ax.tick_params(axis='both', labelsize=7)
    ax.set_xlabel('Prediction Rate', fontsize=8)
    ax.set_ylabel('Precision', fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fileNamePng)
    plt.savefig(fileNamePdf)
    plt.close()

