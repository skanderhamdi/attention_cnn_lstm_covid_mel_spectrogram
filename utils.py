import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.metrics import f1_score, roc_auc_score ,precision_score, recall_score, balanced_accuracy_score

## Function for plotting curves

def my_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average="binary",zero_division=1)
def my_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average="binary")
def my_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average=None)[1]
def my_roc_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred, average=None)
def get_specificity(balanced_acc,recall_arr):
    return 2*(np.asarray(balanced_acc))-(np.asarray(recall_arr))
def get_balanced_acc(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)


def plotConfusionMatrix(y_true,y_pred):
    conf_matrix = confusion_matrix(y_true,y_pred)
    norm_array = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:,np.newaxis]
    group_counts = ["{0:0.0f}".format(value) for value in conf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in norm_array.flatten()]
    labels = [f"{v1}\n\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    df_cm = pd.DataFrame(conf_matrix, range(2), range(2))
    ax = sn.heatmap(df_cm, annot=labels,fmt='', cmap='Greens')
    ax.set_title('CNN model confusion matrix');
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values');
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Non-Likely-COVID-19','Likely-COVID-19'])
    ax.yaxis.set_ticklabels(['Non-Likely-COVID-19','Likely-COVID-19'],va="center")
    plt.show()
    
def plotCurves(title,x,y,curve,histories):
    plt.figure(figsize=(14,8))
    train_all = []
    train_avg = []
    for entry in histories:
        plt.plot(entry[curve],ls='dashed',alpha=0.2,color='tomato')
        train_all.append(entry[curve])
    train_all = np.array(train_all)
    train_avg = np.average(train_all, axis=0)
    plt.plot(train_avg,ls='-',lw=2,label='Average train '+curve,color='tomato')
    val_all = []
    val_avg = []
    for entry in histories:
        plt.plot(entry['val_'+curve],ls='dashed',alpha=0.2,color='darkcyan')
        val_all.append(entry['val_'+curve])
    val_all = np.array(val_all)
    val_avg = np.average(val_all, axis=0)
    plt.plot(val_avg,ls='-',lw=2,label='Average validation '+curve,color='darkcyan')
    plt.title(title)
    plt.ylabel(x)
    plt.xlabel(y)
    plt.legend(loc='best')
    plt.grid()
    plt.show()

def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    printProgressBar(0)
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    print()
    
def plotROCCurve(fpr,tpr,auc,color,label,title):
    plt.figure(figsize=(14,8))
    ax = plt.gca()
    ax.set_facecolor((1.0, 1.0, 1.0))
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth('2')  
    ax.grid(b=True, which='major', color='grey', linestyle='-', alpha=0.3)
    plt.plot(fpr,tpr, lw=2, label= label+' (area = {:.3f})'.format(auc),color = color)
    plt.plot([0, 1], [0, 1], '--',color='grey',label='Random model')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(title)
    legend = plt.legend(loc="best", edgecolor="grey")
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((1, 1, 1, 0.7))
    plt.show()
