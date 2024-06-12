
import numpy as np
import torch
import os
from datetime import datetime

from ArtNeuralNetwork import preprocess, ArtNeuralNetwork

n = ArtNeuralNetwork()
n.load_state_dict(torch.load('ART.pth'))

directory = "datasets/test/"

labels = ["sculpture", "painting"]

num_classes = len(labels)

correct = 0
total = 0

label_count = [0, 0]
label_correct = [0, 0]

for i in range(num_classes):
    dir = directory + labels[i] + '/'
    files = os.listdir(dir)
    for filename in files:
        f = dir + filename
        try:
            img = preprocess(f)
        except:
            continue

        label = i
        label_count[label] += 1

        total += 1
        output = n.forward(img).detach().cpu().numpy()
        print(output)
      
        guess = 0
  
        if output[0] > 0.5:
            guess = 1

        if guess == label:
            correct += 1
            label_correct[label] += 1


print("Accuracy:", correct / total)
print(label_count)
print(label_correct)