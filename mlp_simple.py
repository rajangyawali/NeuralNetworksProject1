import torch
from evaluation import accuracy_precision_recall_f1, plot_confusion_matrix
from dataset import load_mnist, load_mnist_custom_dataset
import config
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utility import test_model, train_model


# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self, input_neurons, output_neurons):
        super(MLP, self).__init__()
        self.multi_layer_percepton = nn.Sequential(
            nn.Linear(input_neurons, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_neurons)
        )
        
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.multi_layer_percepton(x)
        return x

classes = [i for i in range(0,10)]

# Create instance of the MLP, loss function, and optimizer
model = MLP(input_neurons=28*28, output_neurons=len(classes))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

train_dataset, test_dataset, train_loader, test_loader = load_mnist()
train_model(model, train_dataset, train_loader, criterion, optimizer, 'MLP-small')

# Load your trained model
# model = torch.load('output/MLP-small_model_20_epochs.pth')
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("[INFO] Total Number of Parameters : {}".format(total_params))

# Testing Procedure
confusion_matrix = test_model(model, test_loader, num_classes=len(classes))

# # Plot the confusion matrix
plot_confusion_matrix(confusion_matrix.numpy(), classes=classes, model_name="MLP_MNIST_3_vs_8")
accuracy_precision_recall_f1(confusion_matrix.numpy(), num_classes=len(classes), model_name='MLP_MNIST_3_vs_8')

