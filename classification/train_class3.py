import os

import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, resnet34, resnet50

from RSdataset import RSdataset
from utils import plot_image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

batchsz = 32
lr = 1e-3
epochs = 100
weight_name = "resnet18-whurs3.pth"

device = torch.device('cuda')
torch.manual_seed(1234)


train_db = RSdataset('WHURS3', 300, mode='train')
val_db = RSdataset('WHURS3', 300, mode='val')
test_db = RSdataset('WHURS3', 300, mode='test')
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True)
val_loader = DataLoader(val_db, batch_size=batchsz)
test_loader = DataLoader(test_db, batch_size=batchsz)


def get_class_name(labels:list):
    text_labels = ['Beach','Forest', 'Mountain']
    return [text_labels[int(i)] for i in labels]



def make_model(num_class=3):
    pretrained_model = resnet18(pretrained=True)
    model = nn.Sequential(*list(pretrained_model.children())[:-1],  # [b, 512, 1, 1]
                          nn.Flatten(),  # [b, 512, 1, 1] => [b, 512]
                          nn.Linear(512,128),
                          nn.ReLU(),
                          nn.Linear(128, num_class)
                          ).to(device)
    return model


def evalute(model, loader):
    model.eval()
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        # plot_image(x,y,"class")
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            logits = model(x)
            #print("logits: ",logits)
            pred = logits.argmax(dim=1)
        print("y: ", y)
        print("pred: ", pred)
        correct += torch.eq(pred, y).sum().float().item()
    print(f"correct:{correct} total:{total}")
    return correct / total


def train_val():
    model = make_model()
    # x = torch.randn(2, 3, 224, 224)
    # print(model(x).shape)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0, 0
    global_step = 0

    for epoch in range(epochs):

        for step, (x, y) in enumerate(train_loader):

            # x: [b, 3, 224, 224], y: [b]
            x, y = x.to(device), y.to(device)

            model.train()
            logits = model(x)
            loss = criteon(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

        if epoch % 1 == 0:
            val_acc = evalute(model, val_loader)
            print(f"epoch:{epoch} loss:{loss.item()} acc: {val_acc}")
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), weight_name)

    print('best acc:', best_acc, 'best epoch:', best_epoch)

    test_acc = evalute(model, test_loader)
    print('test acc:', test_acc)


def test():
    model = make_model()
    model.load_state_dict(torch.load(weight_name))
    test_acc = evalute(model, test_loader)
    print('test acc:', test_acc)


def test_single_image(img_path):
    print(f"image {os.path.basename(img_path)}")
    resize = 300
    img = Image.open(img_path).convert('RGB')
    fig = plt.figure()
    plt.imshow(img)
    tf = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                        std=[0.229, 0.224, 0.225])
    ])

    img = tf(img).unsqueeze(0).to(device)

    model = make_model()
    model.load_state_dict(torch.load(weight_name))
    model.eval()
    with torch.no_grad():
        pred_class = model(img)
        print(pred_class)
        print(pred_class.argmax(dim=1))
        print(get_class_name(pred_class.argmax(dim=1)))


if __name__ == '__main__':
    #train_val()
    #test()
    test_single_image(r'WHURS19\Mountain\Mountain_50.jpg')
