import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
import time

def prepare_plot(original_image, original_mask, predicted_mask, path):
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))

    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(original_image)
    ax[1].imshow(original_mask)
    ax[2].imshow(predicted_mask)

    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")

    # set the layout of the figure and display it
    figure.tight_layout()
    path = path.split("\\")[-1]
    path = path.replace("train", "result")
    figure.savefig(os.path.join("output", path))




def make_predictions(model, image_path):
    # set model to evaluation mode
    model.eval()
    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        image = cv2.imread(image_path, 0)
        image = image.astype("float32") / 255.0

        # resize the image and make a copy of it for visualization
        image = cv2.resize(image, (128, 128))
        orig = image.copy()

        # find the filename and generate the path to ground truth
        # mask
        filename = image_path.split(os.path.sep)[-1]
        filename = filename.replace("images", "masks")

        ground_truth_path = os.path.join(config.MASK_DATASET_PATH,
            filename)


        # load the ground-truth segmentation mask in grayscale mode
        # and resize it
        ground_truth_mask = cv2.imread(ground_truth_path, 0)
        ground_truth_mask = cv2.resize(ground_truth_mask, (config.INPUT_IMAGE_HEIGHT,
            config.INPUT_IMAGE_HEIGHT))

        # make the channel axis to be the leading one, add a batch
        # dimension, create a PyTorch tensor, and flash it to the
        # current device
        image = np.transpose(image).reshape(1, 128, 128)
        image = np.expand_dims(image, 0)
        # print(image.shape)
        image = torch.from_numpy(image).to(config.DEVICE)
        # print(image.shape)

        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        predicted_mask = model(image).abs_().ceil_()
        # predicted_mask = torch.sigmoid(predicted_mask)
        predicted_mask = predicted_mask.cpu().numpy()
        # filter out the weak predictions and convert them to integers
        # predicted_mask = (predicted_mask > config.THRESHOLD) * 255
        predicted_mask = predicted_mask.astype(np.uint8).reshape(128, 128)

        # prepare a plot for visualization
        prepare_plot(orig, ground_truth_mask, predicted_mask, image_path)


print("[INFO] Loading up test images path ...")
images_path = open(config.TEST_PATH).read().strip().split("\n")
images_path = np.random.choice(images_path, size=10)

print("[INFO] Loading up model...")
model = torch.load(config.MODEL_PATH).to(config.DEVICE)
for path in images_path:
	make_predictions(model, path)