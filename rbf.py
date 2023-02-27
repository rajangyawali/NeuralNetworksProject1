import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from tqdm import tqdm
import config
from dataset import load_mnist, CustomDataset

train_dataset, test_dataset, train_loader, test_loader = load_mnist()

# Get the data
train_data = train_dataset.data
train_labels = train_dataset.targets
train_data = train_data.view(train_data.size(0), -1)

# Define Custom Training Data Loader
train_dataset = {'inputs': train_data, 'labels': train_labels} 
train_dataset = CustomDataset(train_dataset, choice = 'Train')
train_loader = DataLoader( shuffle = True,
                                        dataset = train_dataset, 
                                        batch_size = 10 ) 

# Define KMeans clustering model
kmeans = KMeans(n_clusters=10)

# Fit the model to the data
kmeans.fit(train_data)
clusters = kmeans.cluster_centers_.astype(float)
labels = kmeans.labels_

# print(train_data.shape)
# print(clusters)
# print(clusters.shape)
# print(labels.shape)

class RBFNet(nn.Module):
    def __init__(self, clusters):
        super().__init__()
        self.N = clusters.shape[0]
        self.sigs = nn.Parameter( torch.ones(self.N, dtype=torch.float64)*5, requires_grad=False ) # our sigmas
        self.mus = nn.Parameter( torch.from_numpy(clusters), requires_grad=False ) # our means
        print(self.sigs.shape)
        print(self.mus.shape)
        # our connection to the output layer
        self.lin = nn.Parameter( ((torch.rand(self.N, dtype=torch.float64)-0.5)*2.0)*(1.0/self.N), requires_grad=True)
        self.bias = nn.Parameter( ((torch.rand(1, dtype=torch.float64)-0.5)*2.0)*(1.0/self.N), requires_grad=True)
        print(self.lin.shape)
        print(self.bias.shape)
    def forward(self, x):
        res = torch.zeros(x.shape[0], self.N, dtype=torch.float64)
        for j in range(self.N):
            top = torch.sqrt(torch.sum(torch.pow(x - self.mus[j], 2), dim=1))
            res[:, j] = torch.exp((-0.5) * (torch.pow(top, 2) / torch.pow(self.sigs[j], 2)))
        y_pred = torch.zeros(x.shape[0],dtype=torch.float64)
        for i in range(x.shape[0]): # again, could speed up with matrix math!!!
            y_pred[i] = torch.sigmoid( torch.dot(res[i,:],self.lin) + self.bias )
        return y_pred
        
model = RBFNet(clusters)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Train the model
num_epochs = 200

# initialize a dictionary to store training history
H = {"training_loss": [], "training_accuracy": [], "epochs": []}
# loop over epochs
print("[INFO] Training the network...")
start_time = time.time()

# Train the RBF
for epoch in tqdm(range(num_epochs)):
    training_loss = 0.0
    training_accuracy = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        # _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        # training_accuracy += torch.sum(preds == labels.data)
    avg_training_loss = round(training_loss / len(train_loader), 4)
    # avg_training_accuracy = torch.round((training_accuracy.double() / len(train_dataset)), decimals=4)
    avg_training_accuracy = 0
    print(f'[Epoch {epoch + 1}] Loss: {avg_training_loss} Accuracy: {avg_training_accuracy}')
    # update our training history
    H["training_loss"].append(avg_training_loss)
    H["training_accuracy"].append(avg_training_accuracy)
    H["epochs"].append(epoch + 1)

print('Finished Training')

# display the total time needed to perform the training
end_time = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(end_time - start_time))