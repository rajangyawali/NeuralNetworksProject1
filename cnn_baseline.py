import torch
from evaluation import accuracy_precision_recall_f1, plot_confusion_matrix
from dataset import load_mnist
import config
import torch.nn as nn
import torch.optim as optim
from utility import test_model, train_model



# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self, in_channels, output_classes):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.output_classes = output_classes
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, 
                    stride=1, padding=1 ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, 
                    stride=1, padding=1 ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, 
                    stride=1, padding=1 ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.out = nn.Sequential(
            nn.Linear(64*3*3, 20),
            nn.Linear(20, output_classes)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.out(x)
        return x

# Create instance of the CNN, loss function, and optimizer
model = CNN(in_channels=1, output_classes=10).to(config.DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)

train_dataset, test_dataset, train_loader, test_loader = load_mnist()
classes = [i for i in range(0,10)]

# train_model(model, train_dataset, train_loader, criterion, optimizer, 'CNN_LR_0.1')

# Load your trained model
model = torch.load('output/CNN_LR_0.1_model_40_epochs.pth')

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("[INFO] Total Number of Parameters : {}".format(total_params))
# Testing Procedure
confusion_matrix = test_model(model, test_loader, num_classes=len(classes))

# Plot the confusion matrix
plot_confusion_matrix(confusion_matrix.numpy(), classes=classes, model_name="CNN_LR_0.1")
accuracy_precision_recall_f1(confusion_matrix.numpy(), num_classes=len(classes), model_name="CNN_LR_0.1")