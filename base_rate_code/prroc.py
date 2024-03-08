import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {'Attack':[],
        'tpr':[],
        'fpr':[],
        'prec':[],
        'recall':[],
        'Population':[],
        'pi':[],
}

# These are the datapoints visually read off of Carlini's plot.
# (The middle plot of Figure 9)
dataIn = {
    'Carlini et al.':[
         {'tpr':0.1,'fpr':0.00001},
         {'tpr':0.2,'fpr':0.0001},
         {'tpr':0.35,'fpr':0.001},
         {'tpr':0.5,'fpr':0.01},
         {'tpr':0.75,'fpr':0.1},
         {'tpr':1.0,'fpr':0.25},
         {'tpr':1.0,'fpr':1.0},
    ],
    'Shokri et al.':[
         {'tpr':0.0003,'fpr':0.00001},
         {'tpr':0.002,'fpr':0.0001},
         {'tpr':0.015,'fpr':0.001},
         {'tpr':0.1,'fpr':0.01},
         {'tpr':0.4,'fpr':0.1},
         {'tpr':1.0,'fpr':0.5},
         {'tpr':1.0,'fpr':1.0},
    ],
}
skews = [1,30,240]
pops = ['Balanced','Hospitals','Texas']

for i in range(len(skews)):
    skew = skews[i]
    pop = pops[i]
    P = 1
    N = skew
    for name,tests in dataIn.items():
        for test in tests:
            TPR = test['tpr']
            FPR = test['fpr']
            # prec = TP/TP+FP, recall = FP/N = FP/FP+TN
            # P = TP+FN, N = FP+TN (which I know)
            # TPR = TP/P = TP/TP+FN, FPR = FP/N = FP/FP+TN
            TP = TPR * P
            FP = FPR * N
            prec = TP / (TP+FP)
            recall = FPR
            print(f"{name}: TPR = {TPR}, FPR = {FPR}, skew = {skew}")
            print(f"    prec = {round(100*prec,1)}, recall = {recall}")
            data['Attack'].append(name)
            data['tpr'].append(TPR)
            data['fpr'].append(FPR)
            data['prec'].append(prec)
            data['recall'].append(recall)
            data['Population'].append(f"{pop} 1:{skew}")
            baseline = 1/(1+skew)
            pi = (prec-baseline)/(1-baseline)
            data['pi'].append(pi)

df = pd.DataFrame.from_dict(data)

colors = ['mediumblue','orange']
sns.set_palette(sns.color_palette(colors))
plt.figure(figsize=(4, 3))
ax = sns.lineplot(data=df, x="recall", y="prec",hue='Attack',style='Population')
ax.set(xscale='log')
plt.xlabel('Recall',fontsize=11)
plt.ylabel('Precision',fontsize=11)
ax.legend(loc='upper left', bbox_to_anchor=(1.0,1.0), ncol=1)
plt.ylim(0,1)
plt.xlim(0.00001,1)
plt.grid()
plt.savefig('prroc-prec.png',bbox_inches='tight')

plt.figure(figsize=(4, 3))
ax = sns.lineplot(data=df, x="recall", y="pi",hue='Attack',style='Population')
ax.set(xscale='log')
plt.xlabel('Recall',fontsize=12)
plt.ylabel('Precision Improvement',fontsize=12)
ax.legend(loc='upper left', bbox_to_anchor=(1.0,1.0), ncol=1)
plt.ylim(0,1)
plt.xlim(0.00001,1)
plt.grid()
plt.savefig('prroc-pi.png',bbox_inches='tight')