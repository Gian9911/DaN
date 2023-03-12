import math

import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('all emot.csv', sep="\s+")
c = []
a = []
expressions=[]
pleasure = []
arousal = []
sd = []
for i in range (len(df)):
    plt.Circle((0, 0), 1, fill=False)
    #c.append(df.loc[i].at['Color'])
    #a.append(df.loc[i].at['Alpha'])

    expressions.append(df.loc[i].at['Emozioni'])
    pleasure.append(df.loc[i].at['PleasureMean'])
    arousal.append(df.loc[i].at['ArousalMean'])
    #sd.append(max(math.sqrt(df.loc[i].at['Pleasuresd']),math.sqrt(df.loc[i].at['Arousalsd']))*1000)
    plt.scatter(x=float(df.loc[i].at['PleasureMean']), y=float(df.loc[i].at['ArousalMean']),
                #c=df.loc[i].at['Color'],
                #alpha=df.loc[i].at['Alpha'],
                #s=(max(math.sqrt(df.loc[i].at['Pleasuresd']),math.sqrt(df.loc[i].at['Arousalsd']))*300),
                label=df.loc[i].at['Emozioni'])

for i, txt in enumerate(expressions):
    plt.annotate(txt, (pleasure[i], arousal[i]))

#plt.legend(loc='lower center', title="Classes",ncol=5,fontsize=8)
circle1 = plt.Circle((0, 0), 1, color='black',fill=False)
plt.gca().add_patch(circle1)




plt.xlim(-1,1)
plt.ylim(-1,1)
plt.show()