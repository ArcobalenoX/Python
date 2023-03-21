import  torch
from    torch import optim, nn
import  torchvision
from    torch.utils.data import DataLoader
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from    RSdataset import RSdataset
from    resnet import ResNet18


batchsz = 32
lr = 1e-4
epochs = 1000
weight_name = "e1000lr1e-4.pth"
device = 'cuda'
torch.manual_seed(1234)


train_db = RSdataset('WHURS19-test', 224, mode='train')
val_db = RSdataset('WHURS19-test', 224, mode='val')
test_db = RSdataset('WHURS19-test', 224, mode='test')
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True,)
val_loader = DataLoader(val_db, batch_size=batchsz)
test_loader = DataLoader(test_db, batch_size=batchsz)




def evalute(model, loader):
    model.eval()
    correct = 0
    total = len(loader.dataset)

    for x,y in loader:
        x,y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()

    return correct / total

def train_val():
    torch.cuda.empty_cache()
    model = ResNet18(19).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss().to(device)

    best_acc, best_epoch = 0, 0
    global_step = 0

    for epoch in range(epochs):

        for step, (x,y) in enumerate(train_loader):

            # x: [b, 3, 224, 224], y: [b]
            x, y = x.to(device), y.to(device)
            
            model.train()
            logits = model(x)

            #print(logits.shape,y.shape)            
            loss = criteon(logits, y).to(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            global_step += 1

        print(f"eppoch- {epoch} loss: {loss.item()}")

        if epoch % 1 == 0:
            val_acc = evalute(model, val_loader)
            if val_acc> best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), weight_name)
                print(f"best- {epoch} val_acc: {val_acc}")  



    print('best acc:', best_acc, 'best epoch:', best_epoch)
    model.load_state_dict(torch.load(weight_name))

    test_acc = evalute(model, test_loader)
    print('test acc:', test_acc)


def test_only():
    torch.cuda.empty_cache()
    model = ResNet18(19).to(device)
    model.load_state_dict(torch.load(weight_name))
    test_acc = evalute(model, test_loader)
    print('test acc:', test_acc)



if __name__ == '__main__':
    train_val()
    #test_only()
    
