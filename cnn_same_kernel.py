import torch
from evaluation import accuracy_precision_recall_f1, plot_confusion_matrix
from dataset import load_mnist, load_mnist_custom_dataset, load_fashion_mnist
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
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=28, 
                    stride=1, padding=0 ),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, 
                    stride=1, padding=0 ),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, 
                    stride=1, padding=0 ),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),
        )
        self.out = nn.Sequential(
            nn.Linear(64*1*1, 20),
            nn.Linear(20, output_classes)
        )
    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        x = x.view(x.size()[0], -1)
        x = self.out(x)
        return x

classes = [i for i in range(0,10)]
# classes = [1, 8]

# Create instance of the CNN, loss function, and optimizer
model = CNN(in_channels=1, output_classes=len(classes)).to(config.DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


train_dataset, test_dataset, train_loader, test_loader = load_mnist()
# train_model(model, train_dataset, train_loader, criterion, optimizer, 'CNN_Same_Kernel_Size_Adam__0.001')

# train_dataset, test_dataset, train_loader, test_loader = load_fashion_mnist()
# train_model(model, train_dataset, train_loader, criterion, optimizer, 'CNN_Fashion_MNIST')

# train_dataset, test_dataset, train_loader, test_loader = load_mnist_custom_dataset(labels=classes)
# train_model(model, train_dataset, train_loader, criterion, optimizer, 'CNN_MNIST_1_vs_8')

# Load your trained model
model = torch.load('output/CNN_Same_Kernel_Size_Adam__0.001_model_20_epochs.pth')
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("[INFO] Total Number of Parameters : {}".format(total_params))

# Testing Procedure
confusion_matrix = test_model(model, test_loader, num_classes=len(classes))

# Plot the confusion matrix
plot_confusion_matrix(confusion_matrix.numpy(), classes=classes, model_name="CNN_Same_Kernel_Size_Adam__0.001")
accuracy_precision_recall_f1(confusion_matrix.numpy(), num_classes=len(classes), model_name="CNN_Same_Kernel_Size_Adam__0.001")