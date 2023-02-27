import itertools
import os
import config
import numpy as np
import matplotlib.pyplot as plt
from prettytable import FRAME, PrettyTable
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas



def plot_confusion_matrix(cm, classes, model_name, cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    title = "Confusion Matrix for {} Model with {} Epochs".format(model_name, config.MAX_EPOCHS)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.0f' 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(config.BASE_OUTPUT, title + '.jpg'))


def accuracy_precision_recall_f1(cm, num_classes, model_name):
    total = cm.sum()
    diagonal = cm.trace()
    accuracy = (diagonal / total)

    precision = {}
    recall = {}
    f1 = {}
    tpr = {}
    fpr = {}
    for i in range(cm.shape[0]):
        precision[i] = round(cm[i, i] / cm[:, i].sum(), 3)
        recall[i] = round(cm[i, i] / cm[i, :].sum(), 3)
        f1[i] = round((2 * precision[i] * recall[i] / (precision[i] + recall[i])), 3)
        tpr[i] = round(recall[i], 3)
        fpr[i] = round((cm.sum(0)[i] - cm[i, i]) / (cm.sum() - cm[i, :].sum()), 3)


    print("Accuracy:", round(accuracy, 3))

    table = PrettyTable(align='r', hrules=FRAME, vrules=FRAME)
    
    table.field_names = ["Metric"] + [f"Class {i}" for i in range(num_classes)]
    table.add_row(["Accuracy"] + [accuracy for _ in range(num_classes)])
    table.add_row(["Precision"] + [precision[i] for i in range(num_classes)])
    table.add_row(["Recall"] + [recall[i] for i in range(num_classes)])
    table.add_row(["True PR"] + [tpr[i] for i in range(num_classes)])
    table.add_row(["False PR"] + [fpr[i] for i in range(num_classes)])
    table.add_row(["F1 Score"] + [f1[i] for i in range(num_classes)])
    
    fig = plt.figure(figsize=(20, 10))
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.text(0, 0, str(table), fontsize=14)
    ax.axis('off')
    canvas.draw()
    fig.savefig(os.path.join(config.BASE_OUTPUT, "Precision Recall Table for {} Model with {} Epochs.jpg".format(model_name, config.MAX_EPOCHS)), dpi=100)


