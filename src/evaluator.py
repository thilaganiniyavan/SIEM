# src/evaluator.py
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc(y_true, y_scores, out="../data/roc.png"):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr,tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='lower right')
    plt.savefig(out)
    print("[INFO] ROC saved to", out)
