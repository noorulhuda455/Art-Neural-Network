
import numpy as np
import torch
import os
from datetime import datetime
from PIL import Image, ImageOps

from ArtNeuralNetwork import preprocess, ArtNeuralNetwork


def is_bad_file(f):
    try:
        image = preprocess(f)
        return False
    except:
        return True
    
epochs = 3
stop_at = 1729

n = ArtNeuralNetwork()


print("Start:", datetime.now())

directory = "datasets/train/"

labels = ["sculpture", "painting"]

num_classes = len(labels)

file_lists = []
for label in range(num_classes):
    dir = directory + labels[label] + '/'
    files = os.listdir(dir)
    remove_list = []
    for file in files:
        if is_bad_file(dir + file):
            remove_list.append(file)

    for r in remove_list:
        files.remove(r)

    file_lists.append(files)


for epoch in range(epochs):
    print("Epoch:", epoch)

    for i in range(stop_at):
        for label in range(num_classes):
            dir = directory + labels[label] + '/'
            file_list = file_lists[label]
            file_name = file_list[i]
            
            f = dir + file_name
            print(i, stop_at, f)

            img = preprocess(f)
            
            target = np.zeros(1)
            if label == 1:
                target[0] = 1.0

            n.train(img, target)

        if i % 100 == 0:
            print(i, stop_at)


torch.save(n.state_dict(), 'Art.pth')
print("End:", datetime.now())