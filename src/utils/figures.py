import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_results_cm(y_true: list, y_pred: list):
    names = [("Recall", "Row", "Blues"), ("Precision", "Column", "Greens")]
    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(y_true)
    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for i, (name, axis, color) in enumerate(names):
        normalized = cm.astype("float") / cm.sum(axis = (1-i), keepdims = True)
        sns.heatmap(normalized, annot = True, fmt = ".2f", cmap = color, cbar = True,
                    ax = axs[i], xticklabels = labels, yticklabels = labels, 
                    annot_kws = {"size": 6})
        axs[i].set_title(f"{name} - {axis} Normalized", fontweight ='bold', fontsize=11)
        axs[i].set_xlabel("Predicted Label", fontweight ='bold', fontsize=9)
        axs[i].set_ylabel("True Label", fontweight ='bold', fontsize=9)
        axs[i].tick_params(axis="x", labelsize=8)
        axs[i].tick_params(axis="y", labelsize=8)
        axs[i].collections[0].colorbar.ax.tick_params(labelsize=7)
    plt.show()

def plot_train_test(train_results: list, test_results: list, alpha: float = 0.5, beta: float = 0.5):
    # Best Performance
    F1_test = np.asarray(train_results)
    F1_train = np.asarray(test_results)
    distance_to_ideal = np.abs(F1_train - F1_test) / np.sqrt(2)
    score = alpha * F1_test - beta * distance_to_ideal
    best_model_idx = np.argmax(score)
    max_train, max_test = train_results[best_model_idx], test_results[best_model_idx]
    # Plot
    fig, ax = plt.subplots(1, 1, figsize =(6, 5))
    ax.scatter(train_results, test_results)
    ax.scatter(max_train, max_test, color = "gold", marker = "X", label = "Best model")
    ax.plot([0, 1], [0, 1], linestyle = "--", color = "deepskyblue", label = "Ideal performance($F1_{train} = F1_{test}$)")
    # Labels
    ax.set_ylabel("F1 Score - Test", fontweight ='bold', fontsize=9)
    ax.set_xlabel("F1 Score - Train", fontweight ='bold', fontsize=9)
    ax.set_title("Distribution of F1 Scores", fontweight ='bold', fontsize=11)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize = 8)
    plt.show()
    return best_model_idx