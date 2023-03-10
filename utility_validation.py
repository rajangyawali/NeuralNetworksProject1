import os
import torch
import config
import time 
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_model(model, train_dataset, val_dataset, train_loader, val_loader, criterion, optimizer, name='Model'):
    # initialize a dictionary to store training history
    H = {"training_loss": [], "training_accuracy": [], "val_loss": [], "val_accuracy":[], "epochs": []}
    # loop over epochs
    print("[INFO] Training the network...")
    start_time = time.time()

    # Train the CNN
    for epoch in tqdm(range(config.MAX_EPOCHS)):
        training_loss = 0.0
        training_accuracy = 0.0
        val_loss = 0.0
        val_accuracy = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            training_accuracy += torch.sum(preds == labels.data)
        avg_training_loss = round(training_loss / len(train_loader), 4)
        avg_training_accuracy = torch.round((training_accuracy.double() / len(train_dataset)), decimals=4)
        
        # update our training history
        H["training_loss"].append(avg_training_loss)
        H["training_accuracy"].append(avg_training_accuracy)
        H["epochs"].append(epoch + 1)
        
        # Evaluate the model on the validation set
        model.eval()
        with torch.no_grad():
            val_accuracy = 0
            total = 0
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                total += labels.size(0)
                val_accuracy += (preds == labels).sum().item()
            val_loss += criterion(outputs, labels)
            print(type(val_loss))
            avg_val_accuracy = round((val_accuracy / total), 4)
            avg_val_loss = torch.round(val_loss, decimals=4)
            
        # update our validation history
        H["val_loss"].append(avg_val_loss)
        H["val_accuracy"].append(avg_val_accuracy)
        print(f'[Epoch {epoch + 1}] Training Loss: {avg_training_loss} Training Accuracy: {avg_training_accuracy} Validation Loss: {avg_val_loss} Validation Accuracy: {avg_val_accuracy}')
    print('Finished Training')

    # display the total time needed to perform the training
    end_time = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(end_time - start_time))

    # plot the training loss
    PLOT_PATH = "{} Model Training Curve with {} Epochs.jpg".format(name, config.MAX_EPOCHS)
    print("Plotting the training loss...")
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["epochs"], H["training_loss"], label="Training Loss")
    plt.plot(H["epochs"], H["training_accuracy"], label="Training Accuracy")
    plt.plot(H["epochs"], H["val_loss"], label="Validation Loss")
    plt.plot(H["epochs"], H["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Loss/Accuracy")
    plt.xlabel("Number of Epochs")
    plt.xticks([i for i in range(0, config.MAX_EPOCHS + 2, 4)])
    plt.legend(loc="best")
    plt.savefig(os.path.join(config.BASE_OUTPUT, PLOT_PATH))

    # serialize the model to disk
    MODEL_PATH = "{}_model_{}_epochs.pth".format(name, config.MAX_EPOCHS)
    print(MODEL_PATH)
    torch.save(model, os.path.join(config.BASE_OUTPUT, MODEL_PATH))
    
    
def test_model(model, test_loader, num_classes=10):

    # Initialize confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes)
    model.eval()
    with torch.no_grad():
        
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t, p] += 1
                
    return confusion_matrix