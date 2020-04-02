from gcommand_loader import GCommandLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#the network definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)              #conv layer 3x3 width from 1 to 16
        self.pool = nn.MaxPool2d(2, 2)                #pooling layer 2x2
        self.conv2 = nn.Conv2d(16, 32, 3)             #conv layer 3x3 width from 16 to 32
        self.fc1 = nn.Linear(32 * 38 * 23, 32 * 23)   #fully connected layer from 32 * 38 * 23 to 32 * 23 (removing one dimetion)
        self.fc3 = nn.Linear(32 * 23, 30)             #fully connected layer from 32 * 23 to 30 (to the predictions)

    #operating order
    def forward(self, x):
        x = self.pool(F.celu(self.conv1(x)))
        x = self.pool(F.celu(self.conv2(x)))
        x = x.view(-1, 32 * 38 * 23)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x


#method for loading the training data
def load():
	
    dataset = GCommandLoader('./data/train')

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=True,
        num_workers=20, pin_memory=True)
    return train_loader


#method for loading the test data
def load_test():
    mkdir("./data/test2")
    shutil.move('./data/test', "./data/test2")
    dataset = GCommandLoader('./data/test2')
    return dataset


#loading validation to check at the end
def validate(batch_size_validate):
    dataset = GCommandLoader('./data/valid')

    validation_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size_validate, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)

    correct = 0
    total = 0
    with torch.no_grad():
        for data2 in validation_loader:
            inputs2, labels2 = data2
            inputs2, labels2 = inputs2.to(device), labels2.to(device)

            outputs2 = net(inputs2)
            _, predicted = torch.max(outputs2.data, 1)
            total += labels2.size(0)
            correct += (predicted == labels2).sum().item()

    print('Accuracy of the network on the ' + str(batch_size_validate) + ' Validation audios: %.4f ' % (
            100 * correct / total))
    return correct / total


#printing predictions
def print_predictions():
    dataset = load_test()
    f = open("test_y", "w")
    with torch.no_grad():
        for dat, dat_spec in zip(dataset, dataset.spects):
            temp_loader = torch.utils.data.DataLoader(
                [dat], batch_size=1, shuffle=None, pin_memory=True)
            for data2 in temp_loader:
                inputs2, labels2 = data2
                inputs2, labels2 = inputs2.to(device), labels2.to(device)
                outputs2 = net(inputs2)
                _, predicted = torch.max(outputs2.data, 1)
                print(str(dat_spec[0]) + " " + str(predicted[0].item()), file=f)


if __name__ == '__main__':

    oldValid = 0
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    trainloader = load()
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()

    # SGD Optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs - data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zeroing the gradients
            optimizer.zero_grad()

            # forward then backward then optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


    print('Finished Training')
    validate(5000)
    print("")
    print_predictions()


''' 
OUTPUT:

Finished Training
Accuracy of the network on the 5000 Validation audios: 86.8491

'''