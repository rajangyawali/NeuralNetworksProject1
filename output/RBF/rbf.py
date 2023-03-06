import os
import time 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from tqdm import tqdm
import config
from dataset import load_mnist
from evaluation import accuracy_precision_recall_f1, plot_confusion_matrix

# Load the MNIST training data
train_dataset, test_dataset, train_loader, test_loader = load_mnist()
X_train = train_dataset.data.reshape(-1, 784).double().to(config.DEVICE) / 255.0

# Cluster the data using KMeans with k=10
kmeans = KMeans(n_clusters=10)
kmeans.fit(X_train.cpu().numpy())

# Get the coordinates of the cluster centers
centers = torch.tensor(kmeans.cluster_centers_).to(config.DEVICE)
clusters = kmeans.cluster_centers_.astype(float)


# make network
class RBFNet(nn.Module): 
    def __init__(self, clusters, num_classes):
        super().__init__()
        self.N = clusters.shape[0]
        self.num_classes = num_classes
        # our mean and sigmas for the RBF layer
        self.sigs = nn.Parameter(torch.ones(self.N,dtype=torch.float64)*5, requires_grad=False) 
        self.mus = nn.Parameter(torch.from_numpy(clusters), requires_grad=False)
        self.linear = nn.Linear(self.N, self.num_classes, dtype=torch.float64)

    def forward(self, x):
        res = torch.zeros(x.shape[0], self.N, dtype=torch.float64).to(config.DEVICE)
        for i in range(x.shape[0]): # each data point
            for j in range(self.N): # each cluster
                top = torch.sqrt(((x[i,:]-self.mus[j,:])**2).sum(axis=0))
                res[i,j] = torch.exp((-0.5) * ( torch.pow(top, 2.0) / torch.pow(self.sigs[j], 2.0)))
        
        out = self.linear(res)
        return out

classes = [i for i in range(10)]   
 
# Create instance of the RBF, loss function, and optimizer    
model = RBFNet(clusters, num_classes=len(classes)).to(config.DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# # Train the model
MAX_EPOCHS = 20

# # initialize a dictionary to store training history
# H = {"training_loss": [], "training_accuracy": [], "epochs": []}
# print("[INFO] Training the network...")
# start_time = time.time()

# # Train the Model
# for epoch in tqdm(range(MAX_EPOCHS)):
#     training_loss = 0.0
#     training_accuracy = 0.0
#     for i, (inputs, labels) in enumerate(train_loader, 0):
#         inputs, labels = inputs.view(-1, 784).to(config.DEVICE), labels.to(config.DEVICE)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         _, preds = torch.max(outputs, 1)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         training_loss += loss.item()
#         training_accuracy += torch.sum(preds == labels.data)
#     avg_training_loss = round(training_loss / len(train_loader), 4)
#     avg_training_accuracy = torch.round((training_accuracy.double() / len(train_dataset)), decimals=4)
#     print(f'[Epoch {epoch + 1}] Loss: {avg_training_loss} Accuracy: {avg_training_accuracy}')
#     # update our training history
#     H["training_loss"].append(avg_training_loss)
#     H["training_accuracy"].append(avg_training_accuracy)
#     H["epochs"].append(epoch + 1)

# print('Finished Training')

# # display the total time needed to perform the training
# end_time = time.time()
# print("[INFO] total time taken to train the model: {:.2f}s".format(end_time - start_time))

# # plot the training loss
# PLOT_PATH = "RBF Model_Adam_LR_0.1_50_clusters Training Curve with {} Epochs.jpg".format(MAX_EPOCHS)
# print("Plotting the training loss...")
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(H["epochs"], H["training_loss"], label="Training Loss")
# plt.plot(H["epochs"], H["training_accuracy"], label="Training Accuracy")
# plt.title("Training Loss/Accuracy")
# plt.xlabel("Number of Epochs")
# plt.xticks([i for i in range(0, MAX_EPOCHS + 2, 4)])
# plt.legend(loc="best")
# plt.savefig(os.path.join(config.BASE_OUTPUT, PLOT_PATH))

# # serialize the model to disk
# MODEL_PATH = "RBF_model_Adam_LR0.1_50_clusters{}_epochs.pth".format(MAX_EPOCHS)
# print(MODEL_PATH)
# torch.save(model, os.path.join(config.BASE_OUTPUT, MODEL_PATH))

classes = [i for i in range(0,10)]
num_classes=len(classes)
# Load your trained model
model = torch.load('output/RBF_model_Adam_LR0.1_20_epochs.pth')

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("[INFO] Total Number of Parameters : {}".format(total_params))

# Testing Procedure
# Initialize confusion matrix
confusion_matrix = torch.zeros(num_classes, num_classes)
model.eval()
with torch.no_grad():   
    for inputs, labels in test_loader:
        inputs, labels = inputs.view(-1, 784).to(config.DEVICE), labels.to(config.DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion_matrix[t, p] += 1

# Plot the confusion matrix
plot_confusion_matrix(confusion_matrix.numpy(), classes=classes, model_name="RBF_Adam_LR_0.1_10_clusters")
accuracy_precision_recall_f1(confusion_matrix.numpy(), num_classes=num_classes, model_name="RBF_Adam_LR_0.1_10_clusters")

