import itertools
import os
import argparse
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import seaborn as sn
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from networks.dan import DAN
import tensroflow as tf

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    #print(classes)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    if normalize:
        cm=np.round(cm.astype('float')/cm.sum(axis=1)[:,np.newaxis],2)

        print('Normalized')
    print(cm)
    thresh=cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):

        #plt.text(j,i,cm[i,j],
        plt.text(j, i, str(int((cm[i, j]*100)))+'%',
                 horizontalalignment="center",
                 color="white" if (cm[i,j]>thresh and i<=1 ) else "black")
                 #color =  "black") add i<=1 for train and i!=4 for val

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Image file for evaluation.')

    return parser.parse_args()


class Model():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt','irritated','laugther','nervous','relief','shy']

        self.model = DAN(num_head=4, num_class=13, pretrained=False)

        checkpoint = torch.load('./checkpoints/34_layer_fine_tuned_13_epoch14_acc0.4977.pth',
                                map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect(self, img0):
        img = cv2.cvtColor(np.asarray(img0), cv2.COLOR_RGB2BGR)
        faces = self.face_cascade.detectMultiScale(img)

        return faces

    def fer(self, path):
        img0 = Image.open(path).convert('RGB')

        faces = self.detect(img0)

        if len(faces) == 0:
            return 'null'

        ##  single face detection
        x, y, w, h = faces[0]

        img = img0.crop((x, y, x + w, y + h))

        img = self.data_transforms(img)
        img = img.view(1, 3, 224, 224)
        img = img.to(self.device)

        with torch.set_grad_enabled(False):

            out = self.model(img)
            tf.cast(out)
            _,pred = torch.max(out,1)
            index = int(pred)
            label = self.labels[index]

            return label


if __name__ == "__main__":
    #args = parse_args()
    y_true=[]
    y_pred=[]
    model = Model()

    labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt','irritated','laugther','nervous','relief','shy']


    #image = args.image
    for i in range(len(labels)):

        entries = os.listdir('./affNetNewCLassification/val/'+str(i)+'/')

        path = './affNetNewCLassification/val/'+str(i)+'/'
        for entry in entries:
            #print(entry)
            #print(image)
            #assert os.path.exists(entry), "Failed to load image file."

            label = model.fer(path+str(entry))
            if label != 'null':
                #print(f'emotion label: {label}')
                y_pred.append(label)
                y_true.append(labels[i])



    cm=confusion_matrix(y_true=y_true,y_pred=y_pred,labels=labels)
    print(cm)
    plot_confusion_matrix(cm=cm,classes=labels,normalize=True)
    plot_confusion_matrix(cm=cm,classes=labels,normalize=False)

