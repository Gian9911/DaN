import math
import os
import numpy as np
import pandas as pd
import csv
import shutil

def distance(arousal1,arousal2, pleasure1,pleasure2):

    dist=math.sqrt((arousal1-arousal2)**2 + (pleasure1-pleasure2)**2)
    return dist

def getMinDistance(img_arousal, img_val, img_label):
    dist=100000
    myLabel=img_label
    for row in range(len(df_Clas)):
        if img_label == df_Clas.loc[row].at['Primaria']:
            d = distance(img_arousal,df_Clas.loc[row].at['ArousalMean'],img_val,df_Clas.loc[row].at['PleasureMean'])
            if d < dist:
                dist=d
                myLabel=df_Clas.loc[row].at['Emozioni']
    return myLabel

list=[]

df_Clas=pd.read_csv('classificazione.csv')

df_Aff=pd.read_csv('affectnet copia.csv')

for row in range(len(df_Aff)):#non voglio vederli tutti per il momento, me ne bastano alcuni
    #print('ciao')#291643
    path=str(df_Aff.loc[row].at['img_path'])+str(df_Aff.loc[row].at['name'])
    print(str(row)+'of'+str(len(df_Aff)))
    label = df_Aff.loc[row].at['label']
    name=df_Aff.loc[row].at['name']
    len_name=len(name)
    name_without_jpeg=name[0:len_name-4]
    phase=df_Aff.loc[row].at['phase']+'_set'
    small_phase=df_Aff.loc[row].at['phase']
    aro = float(np.load('./datasets/AfectNet/'+str(phase)+'/annotations/' + str(name_without_jpeg) + '_aro.npy'))
    val = float(np.load('./datasets/AfectNet/'+str(phase)+'/annotations/' + str(name_without_jpeg) + '_val.npy'))
    #print(name+str(aro))
    neeLabelling=getMinDistance(aro, val, label)
    aro_val = [name, path, aro, val, label, neeLabelling,small_phase]
    list.append(aro_val)

#header=[['Nome,Arousal,Valence,OldLabel,NewLabel']]
with open('example.csv', 'w', encoding='UTF8', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Nome', 'Path', 'Arousal', 'Valence', 'OldLabel', 'NewLabel','Phase'])
    writer.writerows(list)
df_example=pd.read_csv('example.csv')
for row in range(len(df_Aff)):#non voglio vederli tutti per il momento, me ne bastano alcuni
   #print(df_example.loc[row].at['Path'])
   #print(df_example.loc[row].at['Phase'])
   #print(df_example.loc[row].at['NewLabel'])
    if df_example.loc[row].at['Phase']=='train':
        shutil.move(df_example.loc[row].at['Path'], './newEmotions_train/'+str(df_example.loc[row].at['NewLabel']))
    else:
        shutil.move(df_example.loc[row].at['Path'], './newEmotions_val/' + str(df_example.loc[row].at['NewLabel']))









