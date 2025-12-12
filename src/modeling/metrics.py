from sklearn.metrics import roc_auc_score, roc_curve, auc, average_precision_score
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
from sklearn.metrics import classification_report

def cal_classification_report(y_true, y_pre, labels, threshold=0.5):

    true_labels, pre_prob, _, cls_labels = preprocess_labels(y_true, y_pre, labels, threshold)
    
    pre_prob = (pre_prob > threshold).astype(int)

    return classification_report(true_labels, pre_prob, target_names=cls_labels)

def cal_multilabel_metrics(y_true, y_pre, labels, threshold=0.5):
    # Convert tensors to numpy and filter out empty classes
    true_labels, pre_prob, _, cls_labels = preprocess_labels(y_true, y_pre, labels, threshold)

    # ---------------- Wanted metrics ----------------

    # -- Average precision score
    macro_avg_prec = average_precision_score(true_labels, pre_prob, average = 'macro')
    micro_avg_prec = average_precision_score(true_labels, pre_prob, average = 'micro')

    # -- AUROC score
    micro_auroc = roc_auc_score(true_labels, pre_prob, average = 'micro')
    macro_auroc = roc_auc_score(true_labels, pre_prob, average = 'macro')
    
    return macro_avg_prec, micro_avg_prec, macro_auroc, micro_auroc

    
def preprocess_labels(y_true, y_pre, labels, threshold = 0.5, drop_missing = True):
    ''' Convert tensor variables to numpy and check the positive class labels. 
    If there's none, leave the columns out from actual labels, binary predictions,
    logits and class labels used in the classification.
    
    :param y_true: Actual class labels
    :type y_true: torch.Tensor
    :param y_pre: Logits of predicted labels
    :type y_pre: torch.Tensor
    
    :return true_labels, pre_prob, pre_binary, labels: Converted (and possibly filtered) actual labels,
                                                       binary predictions and logits

    :rtype: numpy.ndarrays
    '''

    # Actual labels from tensor to numpy
    if isinstance(y_true, np.ndarray):
        true_labels = y_true.astype(np.int32)  
    else:
        true_labels = y_true.cpu().detach().numpy().astype(np.int32)  

    # Logits from tensor to numpy
    if isinstance(y_pre, np.ndarray):
        pre_prob = y_pre.astype(np.float32)  
    else:
        pre_prob = y_pre.cpu().detach().numpy().astype(np.float32)
    
    # ------ One-hot-endcode predicted labels ------

    pre_binary = np.zeros(pre_prob.shape, dtype=np.int32)

    # Find the index of the maximum value within the logits
    likeliest_dx = np.argmax(pre_prob, axis=1)

    # First, add the most likeliest diagnosis to the predicted label
    #pre_binary[np.arange(true_labels.shape[0]), likeliest_dx] = 1

    # Then, add all the others that are above the decision threshold
    other_dx = pre_prob >= threshold

    pre_binary = pre_binary + other_dx
    pre_binary[pre_binary > 1.1] = 1
    pre_binary = np.squeeze(pre_binary) 

    if drop_missing:
        
         # ------ Check the positive class labels ------
    
        # Find all the columnwise indexes where there's no positive class
        null_idx = np.argwhere(np.all(true_labels[..., :] == 0, axis=0))

        # Drop the all-zero columns from actual labels, logits,
        # binary predictions and class labels used in the classification
        if any(null_idx):
            true_labels = np.delete(true_labels, null_idx, axis=1)
            pre_prob = np.delete(pre_prob, null_idx, axis=1)
            pre_binary = np.delete(pre_binary, null_idx, axis=1)
            labels = np.delete(labels, null_idx)

    # There should be as many actual labels and logits as there are labels left
    assert true_labels.shape[1] == pre_prob.shape[1] == pre_binary.shape[1] == len(labels)
    
    return true_labels, pre_prob, pre_binary, labels


def compute_modified_confusion_matrix(labels, outputs):
    '''Compute a modified confusion matrix for multi-class, multi-label tasks.
    
    :param labels: Actual class labels
    :type labels: numpy.ndarray
    :param outputs: One-hot-encoded predicted class labels
    :type outputs: numpy.ndarray
    
    :return A: Multi-class, multi-label confusion matrix
    :rtype: numpy.ndarray
    
    '''

    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0/normalization
    return A

def roc_curves_plot(y_true, y_pre, labels):
    roc_auc, fpr, tpr, cls_labels = roc_curves(y_true, y_pre, labels)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 5))
    fig.suptitle('ROC Curves')

    # Plotting micro-average and macro-average ROC curves
    ax1.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))

    ax1.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]))

    # Plotting ROCs for each class
    for i in range(len(cls_labels)):
        ax2.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(cls_labels[i], roc_auc["labels"][cls_labels[i]]))

    # Adding labels and titles for plots
    ax1.plot([0, 1], [0, 1], 'k--'); ax2.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlim([0.0, 1.0]); ax2.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05]); ax2.set_ylim([0.0, 1.05])
    ax1.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax2.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax1.legend(loc="lower right", prop={'size': 8}); ax2.legend(loc="lower right", prop={'size': 6})
    
    fig.tight_layout()
    return roc_auc, fpr, tpr, cls_labels, fig

def roc_curves_notebook(y_true, y_pre, labels, epoch=None):
    roc_auc, _, _, cls_labels, fig = roc_curves_plot(y_true, y_pre, labels)
    plt.show()

def roc_curves_save(y_true, y_pre, labels, save_path, epoch=None):
    roc_auc, _, _, cls_labels, fig = roc_curves_plot(y_true, y_pre, labels)
    name = "roc-e{}.png".format(epoch) if epoch else "roc-test.png"
    plt.savefig(os.path.join(save_path, name), bbox_inches = "tight")
    plt.close(fig) 

def roc_curves(y_true, y_pre, labels):
    '''Compute and plot the ROC Curves for each class, also macro and micro.
    
    :param y_true: Actual labels
    :type y_true: torch.Tensor
    :param y_pred: Logits of predicted labels
    :type y_pred: torch.Tensor
    :param labels: Class labels used in the classification as SNOMED CT Codes
    :type labels: list
    '''

    # Convert tensors to numpy and filter out empty classes
    true_labels, pre_prob, _, cls_labels = preprocess_labels(y_true, y_pre, labels, drop_missing=True)
    
    fpr, tpr, roc_auc = dict(), dict(), {'labels':{}}
    # AUROC, fpr and tpr for each label
    for i in range(len(cls_labels)):
        fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], pre_prob[:, i])
        #roc_auc['labels'][cls_labels[i]] = auc(fpr[i], tpr[i])
        roc_auc['labels'][cls_labels[i]] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true_labels.ravel(), pre_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Interpolate all ROC curves at these points to compute macro-average ROC area
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.zeros_like(fpr_grid)
    for i in range(len(cls_labels)):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average the mean TPR and compute AUC
    mean_tpr /= len(cls_labels)
    
    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr

    roc_auc["fpr_macro"],roc_auc['tpr_macro'],roc_auc['fpr_micro'],roc_auc['tpr_micro'] =\
    fpr["macro"],tpr["macro"],fpr["micro"], tpr["micro"]

    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
 
    return roc_auc, fpr, tpr, cls_labels

def train_loss_progression(hist, plot_dir="."):
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("Training loss")
    x = np.arange(len(hist["train_loss"]))
    plt.plot(x, hist["train_loss"])
    plt.savefig(os.path.join(plot_dir, "training_loss.png"), bbox_inches = "tight")
    plt.close()

def fl_plot_progression(progr_stats, key="train_loss", guides=[]):
    collab_names = progr_stats.keys()
    for k in collab_names:
        if k in progr_stats and key + "_x" in progr_stats[k]:
            y = progr_stats[k][key + "_y"]
            x = progr_stats[k][key + "_x"]
            plt.plot(x, y, label=k)
    for guide in guides:
        plt.axhline(y=guide, color='k', linestyle="--")

def fl_train_progression(progr_stats, plot_dir=".", key="train_loss", label="Training loss", guides=[]):
    fl_plot_progression(progr_stats, key, guides=guides)
    plt.ylabel(label)
    if "loss" in key:
        plt.legend(loc="upper right")
    else:
        plt.legend(loc="lower right")
    plt.xlabel("federated round")
    plt.axhline(y=0, color='k')
    plt.savefig(os.path.join(plot_dir, f"{key}.png"), bbox_inches = "tight")
    plt.close()

if __name__ == '__main__':

    y_actual = torch.Tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

    y_prob = torch.Tensor([[0.0000e+00, 1.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00,
         1.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00,
         1.0000e+00, 1.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00],
        [7.5481e-34, 1.0000e+00, 1.0000e+00, 3.5030e-19, 7.4219e-26, 1.0000e+00,
         1.0000e+00, 1.4667e-36, 1.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00,
         1.0000e+00, 1.0000e+00, 2.0872e-25, 1.0000e+00, 1.0000e+00],
        [4.9990e-28, 1.0000e+00, 1.0000e+00, 8.5356e-28, 3.5239e-24, 1.0000e+00,
         1.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 1.0000e+00, 8.3170e-01,
         1.0000e+00, 1.0000e+00, 1.2724e-32, 1.0000e+00, 1.0000e+00],
        [1.8659e-09, 8.7257e-06, 1.0000e+00, 3.9260e-33, 8.4741e-31, 1.0000e+00,
         1.0000e+00, 9.1425e-25, 1.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00,
         1.0000e+00, 6.5847e-24, 0.0000e+00, 1.0000e+00, 1.0000e+00],
        [1.3829e-29, 1.0000e+00, 1.0000e+00, 6.8302e-31, 6.7060e-20, 1.0000e+00,
         1.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00,
         1.0000e+00, 1.0000e+00, 6.5582e-24, 1.0000e+00, 1.0000e+00],
        [3.7383e-10, 2.8861e-02, 1.0000e+00, 1.6027e-24, 1.4673e-14, 1.0000e+00,
         1.0000e+00, 4.0820e-28, 1.0000e+00, 0.0000e+00, 1.0000e+00, 4.6228e-02,
         1.0000e+00, 9.0010e-05, 8.0360e-18, 1.0000e+00, 1.0000e+00],
        [1.1432e-22, 9.9910e-01, 1.0000e+00, 1.1678e-17, 4.7706e-29, 1.0000e+00,
         1.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00,
         1.0000e+00, 1.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00],
        [2.4080e-23, 1.0000e+00, 1.0000e+00, 8.2847e-15, 3.4797e-17, 1.0000e+00,
         1.0000e+00, 4.6156e-36, 1.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00,
         1.0000e+00, 1.0000e+00, 8.2097e-26, 1.0000e+00, 1.0000e+00]])
    
    labels = ['426783006', '426177001', '164934002', '427393009', '713426002',
       '427084000', '59118001', '164889003', '59931005', '47665007',
       '445118002', '39732003', '164890007', '164909002', '270492004',
       '251146004', '284470004']

    cal_multilabel_metrics(y_actual, y_prob, labels, threshold=0.5)
