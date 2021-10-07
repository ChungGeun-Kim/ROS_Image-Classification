import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary as summary_
import os
from ResNet import ResNet50

transform_train = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

transform_test = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_dataset = torchvision.datasets.ImageFolder(root='/home/kcg/pytorch_ex/data/hand_data5/dataset/train', 
                                                 transform=transform_train)

test_dataset = torchvision.datasets.ImageFolder(root='/home/kcg/pytorch_ex/data/hand_data5/dataset/test', 
                                                transform=transform_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=16,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=16,
                                          shuffle=True)


net = ResNet50(10).to('cuda')
summary_(net,(3,64,64), batch_size=16)
file_name = 'resnet50_test10.pth'

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.01, patience=5)

EPOCHS = 100
for epoch in range(EPOCHS):
    losses = []
    running_loss = 0
    for i, inp in enumerate(train_loader):
        inputs, labels = inp
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
    
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if i%10 == 0 and i > 0:
            print(f'Loss [{epoch+1}, {i}](epoch, minibatch): ', running_loss / 100)
            running_loss = 0.0

    avg_loss = sum(losses)/len(losses)
    scheduler.step(avg_loss)
            
print('Training Done')

correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = net(images)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(predicted)
        print(labels,'\n')

print('Accuracy on test images: ', 100*(correct/total), '%')

if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
torch.save(net.state_dict(), './checkpoint/' + file_name)
print('Model Saved!')        
