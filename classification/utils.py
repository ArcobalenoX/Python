from    matplotlib import pyplot as plt
import  torch
from    torch import nn
import math
class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

def get_class_name(labels:list):
    text_labels = ['Airport', 'Beach', 'Bridge', 'Commercial', 'Desert', 
    'Farmland', 'FootballField', 'Forest','Industrial', 'Meadow', 
    'Mountain', 'Park', 'Parking', 'Pond', 'Port',
    'RailwayStation', 'Residential', 'River', 'Viaduct']
    return [text_labels[int(i)] for i in labels]

def plot_image(img, label, name):

    num = len(label)
    r = int(math.sqrt(num))
    label = get_class_name(label)
    fig = plt.figure()
    for i in range(num):
        plt.subplot(r+1, int(num/r), i + 1)
        plt.tight_layout()
        #plt.imshow(img[i][0]*0.3081+0.1307, cmap='gray', interpolation='none')
        plt.imshow(img[i][0]*0.3081+0.1307,cmap='gray',)
        plt.title(f"{name}: {label[i]}")
        plt.xticks([])
        plt.yticks([])
    plt.show()