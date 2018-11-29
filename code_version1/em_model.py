import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
import argparse
import matplotlib.pyplot as plt
import os
import time
import shutil
import numpy as np

class EmNet(nn.Module):
    # Define the network structure
    # classes: list of target classes
    def __init__(self, classes):
        super(EmNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 2 * 2, 3072),
            nn.ReLU(inplace=True),
            nn.Linear(3072, len(classes)),
        )

    # Forward process
    # x is a 48x48 2d gray scale image
    def forward(self, y):
        # print(y)
        x = torch.tensor(y)
        x = self.features(x)
        # print(x.size())
        x = x.view(x.size(0), 128 * 2 * 2)
        x = self.classifier(x)
        # print(x)
        return x

class EmModel:
    def __init__(self, dataDir, modelDir):
        self.dataDir = dataDir
        self.modelDir = modelDir
        # Initialize target classes and EmNet model
        self.classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'neutral']
        self.net = EmNet(self.classes)

        # Initialize training parameters
        self.epochs = 50
        self.batch_size = 1
        self.learn_rate = 0.0001
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.classifier.parameters(), lr=self.learn_rate)

        # Load data set
        trainDir = os.path.join(dataDir, 'train')

        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        train_dataset = datasets.ImageFolder(
            trainDir,
            transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                normalize,
        ]))
        
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = self.batch_size, shuffle=True, num_workers = 0)
        print(train_dataset.classes)

        self.check_point_file = os.path.join(modelDir, 'checkpoint.tar')
        if os.path.isfile(self.check_point_file):
            print('Found check point file:', self.check_point_file)
            copy = torch.load(self.check_point_file)
            self.end = copy['epoch']
            self.best_acc = copy['best_acc']
            self.net.load_state_dict(copy['model'])
            self.optimizer.load_state_dict(copy['optimizer'])
            self.loss_list = copy['loss_list']
            self.acc_list = copy['acc_list']
            plt.plot(list(range(1, len(self.loss_list)+1)), self.loss_list, label='EmNet')
            plt.legend()
            plt.title('Training Error vs Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Training Error')
            plt.savefig('error.png')
            plt.clf()
            
            plt.plot(list(range(1, len(self.acc_list)+1)), self.acc_list, label='EmNet')
            plt.legend()
            plt.title('Prediction Accuracy vs Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.savefig('accuracy.png')
            plt.clf()
        else:
            self.end = -1
            self.best_acc = 0
            self.loss_list = []
            self.acc_list = []

    # inputs is a batch of data (4d tensor)
    # Return the index of predicted class
    def predict(self, inputs):
        outputs = self.net(inputs)
        prob, label = torch.max(outputs, 1)
        '''
        print('prob', prob)
        print('label', label)
        '''
        '''
        # If probability is lower than 0.5
        # Predict it as neutral
        label[prob < 0.5] = 7
        '''
        return label

    # img is a 2d tensor
    # Return the name of predicted class
    def forward(self, img):
        inputs = torch.unsqueeze(img.type(torch.FloatTensor), 0)
        label = predict(inputs)
        result = self.classes[self.classes[label[0]]]
        print('Predict result', result)
        return result

    def train(self):
        def save_model(state, better, f=self.check_point_file):
            torch.save(state, f)
            if better:
                shutil.copyfile(f, os.path.join(self.modelDir, 'saved_model.tar'))

        # Return accuracy rate
        def test():
            total = self.batch_size * len(self.train_loader)
            correct = 0
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data
                labels = labels.type(torch.FloatTensor)
                # print(labels)
                pred = self.predict(inputs)
                pred = pred.type(torch.FloatTensor)
                # print(pred)
                check = torch.eq(pred, labels)
                check.double()
                correct += torch.sum(check).item()

            return correct / total

        print('Training starts:', time.ctime(time.time()))
        
        for epoch in range(self.end + 1, self.epochs):  
            # loop over the dataset multiple times
            print('Training epoch:', epoch)
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                # get the inputs
                inputs, labels = data
                # print(labels)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            running_loss /= len(self.train_loader)
            print('[Epoch %d] loss: %.3f' % (epoch, running_loss))
            self.loss_list.append(running_loss)

            acc = test()
            better = False
            if (acc > self.best_acc):
                self.best_acc = acc
                better = True

            print('Accuracy:', acc)
            self.acc_list.append(acc)

            print('Save check point at epoch', epoch)
            state = {
                    'epoch': epoch,
                    'best_acc': self.best_acc,
                    'model': self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'loss_list': self.loss_list,
                    'acc_list': self.acc_list,
                    'classes': self.classes,
                    }
            save_model(state, True)

        print('Training ends:', time.ctime(time.time()))

model = EmModel('../face_data', 'model')
model.train()
