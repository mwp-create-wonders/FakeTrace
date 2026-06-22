import torch
import numpy as np
from copy import deepcopy
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score


def find_best_threshold(y_true, y_pred):
    best_acc = 0.0
    best_thres = 0.5

    for thres in np.unique(y_pred):
        temp = (y_pred >= thres).astype(np.int32)
        acc = (temp == y_true).mean()
        if acc >= best_acc:
            best_acc = acc
            best_thres = thres

    return best_thres


def calculate_acc(y_true, y_pred, thres):
    pred_label = (y_pred > thres).astype(np.int32)

    r_acc = accuracy_score(y_true[y_true == 0], pred_label[y_true == 0])
    f_acc = accuracy_score(y_true[y_true == 1], pred_label[y_true == 1])
    acc = accuracy_score(y_true, pred_label)

    return r_acc, f_acc, acc


def validate(model, loader, device=None, find_thres=True):
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    with torch.no_grad():
        y_true, y_pred = [], []

        print("Number of batches: %d" % len(loader))
        print("Number of samples: %d" % len(loader.dataset))

        for img, label in loader:
            img = img.to(device)
            logits = model(img)
            probs = torch.sigmoid(logits).flatten()

            y_pred.extend(probs.cpu().tolist())
            y_true.extend(label.flatten().cpu().tolist())

    y_true = np.array(y_true, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.float32)

    ap = None
    auc = None
    if len(np.unique(y_true)) == 2:
        ap = average_precision_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)

    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)

    if not find_thres:
        return ap, auc, r_acc0, f_acc0, acc0

    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)

    return ap, auc, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres