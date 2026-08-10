from sklearn.metrics import auc
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
import os
import psutil
from scipy import stats
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from torchvision.datasets import CIFAR10
from fvcore.nn import FlopCountAnalysis
from tqdm import tqdm
from functools import partial
from utils import load_data_samples, load_model, IMG_SIZE
from explanations import XAI_METHODS, NUMBER_OF_ITERATIONS
from perturbation_pipeline import PERTURBATION_RATIOS

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def average_AUC_scores(perturbation_scores_pth="./perturbation_scores", methods=list(XAI_METHODS.keys())):
    """
    Computes the average Deletion/Insertion AUC scores for each XAI method and prints the results. 
    
    Args:
        perturbation_scores_pth (str): Path to the directory containing the perturbation scores.
        methods (list): List of XAI methods for which to compute average AUC scores.
    """
    print("\n--- Average Deletion/Insertion AUC ---")
    for m in methods:
        print(f"\n{m}:")
        scores_path = f"{perturbation_scores_pth}/{m}"
        deletion_auc = np.load(f"{scores_path}/feature_deletion_auc.npy")
        insertion_auc = np.load(f"{scores_path}/feature_insertion_auc.npy")

        # Average per-instance AUC over each iteration before computing total average AUC
        avg_del_scores = np.mean(deletion_auc, axis=0)        
        print(f"Avg Feature Deletion AUC={np.mean(avg_del_scores):.3f} {chr(177)}{np.std(avg_del_scores):.4f}")
        avg_ins_scores = np.mean(insertion_auc, axis=0)
        print(f"Avg Feature Insertion AUC={np.mean(avg_ins_scores):.3f} {chr(177)}{np.std(avg_ins_scores):.4f}")

def plot_avg_confidence_scores(scores, ratios=PERTURBATION_RATIOS, **config):
    """
    Helper function to plot the average model confidence scores vs. perturbation ratios for each XAI method. 
    
    Args:
        scores (dict): Dictionary containing the average model confidence scores for each XAI method.
        ratios (list): List of perturbation ratios.
        **config: Additional plotting configuration options.
    """
    # Parse plotting configuration
    mode = config.get('mode', "Deletion")
    save_path = config.get('save_path', None)
    save_only = config.get('save_only', False)

    fig, ax = plt.subplots(figsize=(10,5))
    colors = mpl.color_sequences['tab10']
    # colors = mpl.color_sequences['Dark2']
    # colors = mpl.color_sequences['Pastel1']
    # colors = plt.rcParams['color.sequences']['tab10']
    
    x = 100 * ratios
    hatches = ['/', '\\', '///']
    for i, m in enumerate(scores.keys()):
        ax.plot(x, scores[m], linewidth=2, color=colors[i], label=m)
        ax.fill_between(x, scores[m], color=colors[i], alpha=0.2)  # alpha controls transparency
        # ax.fill_between(x, scores[m], facecolor="none", edgecolor=colors[0], hatch=hatches[i], alpha=0.1)

    plt.title(f"Model Confidence (Feature {mode})", fontsize=18)
    xlabel = "Deletion ratio (%)" if mode == "Deletion" else "Insertion ratio (%)"
    plt.xlabel(xlabel, fontsize=16)
    plt.xticks(fontsize=14)
    plt.ylabel("Avg Confidence", fontsize=16)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14)

    # Save the plot for later analysis
    if save_path is not None:
        plt.savefig(save_path)
    
    if save_only:
        plt.close()
        return
    
    plt.show()

def average_confidence_scores(perturbation_scores_pth="./perturbation_scores", methods=list(XAI_METHODS.keys())):
    """
    Computes the average model confidence scores vs. perturbation ratios for each XAI method and plots the results.
    
    Args:
        perturbation_scores_pth (str): Path to the directory containing the perturbation scores.
        methods (list): List of XAI methods for which to compute average confidence scores.
    """
    print("\n--- Average model confidence vs. perturbation ratios ---")
    avg_del_confidence = {m: None for m in methods}
    avg_ins_confidence = {m: None for m in methods}

    # Compute average model perturbation confidence for each XAI method
    for m in methods:
        del_confidence = np.load(f"{perturbation_scores_pth}/{m}/deletion_confidence.npy")
        avg_del_confidence[m] = np.mean(del_confidence, axis=0)
        ins_confidence = np.load(f"{perturbation_scores_pth}/{m}/insertion_confidence.npy")
        avg_ins_confidence[m] = np.mean(ins_confidence, axis=0)

    # Plot the average confidence scores vs perturbation ratios
    print('Generating plots...')
    plot_avg_confidence_scores(avg_del_confidence, mode='Deletion')
    plot_avg_confidence_scores(avg_ins_confidence, mode='Insertion')

def explanation_latency(explanations_path="./explanations",
                        methods=list(XAI_METHODS.keys()),
                        correct_indices_pth="./ckpts/correct_preds_indices.npy"):
    """
    Computes the average explanation latency for each XAI method. 
    
    Args:
        explanations_path (str): Path to the directory containing the generated explanations.
        methods (list): List of XAI methods for which to compute average explanation latency.
        correct_indices_pth (str): Path to the file containing the indices of correctly predicted samples.
    """
    print("\n--- Average Explanation Latency ---")
    correct_pred_idx = np.load(correct_indices_pth)
    N = len(correct_pred_idx)
    for m in methods:
        print(f"\n{m}:")
        durations = [np.load(f"{explanations_path}/{m}/duration_{i}.npy") for i in range(1, NUMBER_OF_ITERATIONS+1)]
        avg_duration_per_sample = np.mean(durations) / N
        print(f"Latency = {1e3 * avg_duration_per_sample:.2f} ms per image sample")
        print(f"Throughput = {1/avg_duration_per_sample:.2f} images/sec")

def peak_vram():
    """ Computes the peak VRAM usage for each XAI method. """
    print("\n--- Peak VRAM ---")

    # Load the pre-trained model
    model = load_model(model_ckpt="ckpts/rn18_cifar10.ckpt")

    # Define the target layer for Grad-CAM
    target_layers = [model.layer4[-1]]  # Last layer of the last block in ResNet18

    # Compute max VRAM usage for each XAI method
    for name, config in XAI_METHODS.items():
        # Force tqdm to stop rendering for every LIME explanation
        tqdm.__init__ = partial(tqdm.__init__, disable=True)

        if name == 'GradCAM':
            explainer = config['ctor'](model, target_layers=target_layers)
        else:
            explainer = config['ctor'](model)

        dummy_image = np.random.randn(1, IMG_SIZE, IMG_SIZE, 3) if name == "LIME" else torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        explanation = explainer.explain(dummy_image, labels=torch.tensor([3]))  # Assuming label 3 for cat (doesn't matter)
        peak = torch.cuda.max_memory_allocated()
        additional_vram = (peak - before) / (1024 ** 3)
        print(f"\n{name}: {additional_vram:.3f} GB")

def ram_usage():
    """ Computes the CPU memory usage for each XAI method. """
    print("\n--- Memory Usage ---")

    # Load the pre-trained model
    model = load_model(model_ckpt="ckpts/rn18_cifar10.ckpt")

    # Define the target layer for Grad-CAM
    target_layers = [model.layer4[-1]]  # Last layer of the last block in ResNet18

    # Compute max VRAM usage for each XAI method
    for name, config in XAI_METHODS.items():
        # Force tqdm to stop rendering for every LIME explanation
        tqdm.__init__ = partial(tqdm.__init__, disable=True)

        if name == 'GradCAM':
            explainer = config['ctor'](model, target_layers=target_layers)
        else:
            explainer = config['ctor'](model)

        dummy_image = np.random.randn(1, IMG_SIZE, IMG_SIZE, 3) if name == "LIME" else torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

        process = psutil.Process(os.getpid())
        before = process.memory_info().rss
        explanation = explainer.explain(dummy_image, labels=torch.tensor([3]))  # Assuming label 3 for cat (doesn't matter)
        after = process.memory_info().rss
        ram_used = (after - before) / (1024 ** 2)
        print(f"\n{name}: {ram_used:.2f} MB")

def estimate_gflops():
    """ 
    Estimates the GFLOPS for each XAI method. The GFLOPS for the backward pass is
    estimated as 2x the GFLOPS for the forward pass.
    """
    print("\n--- Average FLOPS ---")

    # Forward/Backward pass counts for each XAI method
    # (taken from each method's default configuration in their respective Python class modules)
    n_fwd_bwd_passes = {"GradCAM": (1, 1), "IntegratedGradients": (50, 50), "LIME": (500, 0)}

    # Measure GFLOPS for a model forward pass
    dummy_img = torch.randn(1, 3, 224, 224, device=Device)
    model = load_model(model_ckpt="ckpts/rn18_cifar10.ckpt")
    model.to(Device)
    flops = FlopCountAnalysis(model, dummy_img)
    F_forward = flops.total() / 1e9  # GFLOPS
    F_backward = 2 * F_forward  # standard estimation
    print(f"Forward GFLOPs: {F_forward:.3f}, Backward GFLOPs: {F_backward:.3f}\n")

    # Calculate GFLOPS for each XAI method
    for m, counts in n_fwd_bwd_passes.items():
        N_fwd, N_bwd = counts
        gflops = N_fwd * F_forward + N_bwd * N_bwd * F_backward
        print(f"{m}: {gflops:.3f} GFLOPS")

def plot_metrics_by_class(_metrics, _class_labels, _mode='Deletion'):
    """
    Helper function to plot Average Deletion/Insertion AUC by class label and for each XAI method.
    
    Args:
        _metrics (dict): Dictionary containing the average AUC scores for each XAI method and class label.
        _class_labels (list): List of class labels.
        _mode (str): Mode of perturbation, either 'Deletion' or 'Insertion'.
    """
    # Bar Plot of the metrics
    fig, ax = plt.subplots(figsize=(15, 5))
    x = np.arange(1, len(_class_labels)+1)
    width = 0.25
    offset = -0.5
    colors = ['navy', 'cornflowerblue', 'lightskyblue']
    for i, (k, v) in enumerate(_metrics.items()):
        perturbation_auc_by_class = [_metrics[k][c] for c in _class_labels]
        ax.bar(x + offset * width, height=perturbation_auc_by_class, width=width, color=colors[i], label=k)
        offset += 1.0
    ax.set_xticks(x)
    ax.set_xticklabels(_class_labels)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f'Feature {_mode} AUC by class', fontsize=18)
    plt.xlabel('Class', fontsize=16)
    plt.ylabel(f'AUC', fontsize=16)
    plt.legend(loc='upper right')
    plt.show()

def correct_preds_barplot(_correct_preds, _class_labels):
    """
    Plots a bar chart of the counts of correctly predicted samples for each class. 
    
    Args:
        _correct_preds (list): List of class indices for the correctly predicted samples.
        _class_labels (list): List of class labels.
    """
    class_counts = np.unique(_correct_preds, return_counts=True)[1]
    print('Correct Prediction Counts:')
    for c, count in zip(_class_labels, class_counts):
        print(f"{c}: {count}")
    fig, ax = plt.subplots()
    x = np.arange(1, len(_class_labels)+1)
    ax.bar(x, height=class_counts)
    ax.set_xticks(x)
    ax.set_xticklabels(_class_labels)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.title('Correctly Predicted Class Counts', fontsize=16)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel(f'# correct', fontsize=14)
    plt.show()

def auc_by_classes(perturbation_scores_pth="./perturbation_scores",
                   methods=list(XAI_METHODS.keys())):
    """
    Computes and plots the Average Deletion/Insertion AUC by class label for each XAI method.

    Args:
        perturbation_scores_pth (str): Path to the directory containing the perturbation scores.
        methods (list): List of XAI methods for which to compute average AUC.
    """
    print("\n--- Deletion/Insertion AUC by Classes ---")

    # Load the dataset class names and the correctly predicted test samples
    test_dataset = CIFAR10(root='data', train=False)
    class_labels = test_dataset.classes
    correct_samples = load_data_samples()
    
    y_correct = [_[1] for _ in correct_samples.dataset]

    # Visualize how the model performed for each class
    correct_preds_barplot(y_correct, class_labels)

    # Load the deletion/insertion AUC score for each XAI method and average over the iterations.
    avg_deletion_auc = {m: np.mean(np.load(f"{perturbation_scores_pth}/{m}/feature_deletion_auc.npy"), axis=0) for m in methods}
    avg_insertion_auc = {m: np.mean(np.load(f"{perturbation_scores_pth}/{m}/feature_insertion_auc.npy"), axis=0) for m in methods}
    avg_auc_scores = {'Deletion': avg_deletion_auc, 'Insertion': avg_insertion_auc}

    # Compute average AUC by class for each XAI method
    metrics_by_class = {'Deletion': {m: {c: 0.0 for c in class_labels} for m in methods},
                        'Insertion': {m: {c: 0.0 for c in class_labels} for m in methods}}
    for mode, auc in avg_auc_scores.items():
        for i, c in enumerate(class_labels):
            class_idx = [idx for idx, y in enumerate(y_correct) if y == i]
            for m in methods:
                class_auc = auc[m][class_idx]
                metrics_by_class[mode][m][c] = np.mean(class_auc)

    # Plot the results
    for mode in metrics_by_class.keys():
        plot_metrics_by_class(metrics_by_class[mode], class_labels, mode)

def statistical_significance(perturbation_scores_pth="./perturbation_scores", methods=list(XAI_METHODS.keys()), alpha=0.05):
    """
    Determines the statistical significance of the Average Deletion/Insertion AUC results.

    Args:
        perturbation_scores_pth (str): Path to the directory containing the perturbation scores.
        methods (list): List of XAI methods for which to compute average AUC.
        alpha (float): Significance level for the statistical tests.
    """
    deletion_auc = {m: np.load(f"{perturbation_scores_pth}/{m}/feature_deletion_auc.npy") for m in methods}
    insertion_auc = {m: np.load(f"{perturbation_scores_pth}/{m}/feature_insertion_auc.npy") for m in methods}
    results = {'Deletion': deletion_auc, 'Insertion': insertion_auc}

    print("\n" + "="*85 + "\n")
    print(f"Determining statistical significance of the Average Deletion/Insertion AUC results...")
    print(f"--- Pre-requisites: Shapiro-Wilk Test and Levene's Test ---")

    # Shapiro-Wilk Test (Testing for Normality)
    average_auc = {'Deletion': [], 'Insertion': []}
    for m in methods:
        for r in results.keys():
            avg_perturbation_auc = np.mean(results[r][m], axis=1)  # average AUC over each test sample
            shapiro_group = stats.shapiro(avg_perturbation_auc)
            pval = shapiro_group.pvalue
            if pval < alpha:
                print(f"WARNING: Shapiro-Wilk Test failed: {m} {r} AUC: p-value = {shapiro_group.pvalue:.4f}")
            average_auc[r].append(avg_perturbation_auc)

    # Levene's Test (Testing for Homogeneity of Variance)
    for k, v in average_auc.items():
        levene_stat, levene_p = stats.levene(*v)
        if levene_p < alpha:
            print(f"WARNING: Levene's Test failed for {k} AUC: Statistic = {levene_stat:.4f}, p-value = {levene_p:.4f}")

    # Perform One-Way ANOVA Test and Tukey HSD
    for k, v in average_auc.items():
        f_stat, p_value = f_oneway(*v)

        print(f"\n--- ANOVA Test ({k} AUC) ---")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        if p_value < alpha:
            print("Result: Statistically significant (Reject Null Hypothesis)")
        else:
            print("Result: Not statistically significant (Fail to Reject Null Hypothesis)")

        # Tukey HSD
        print(f"\n--- Tukey HSD ({k}) ---")
        num_iterations = len(v)
        num_ts_samples = len(v[0])
        group_names = [m for m in methods for _ in range(num_ts_samples)]
        v_concat = np.array(v).reshape((num_iterations * num_ts_samples, -1))
        tukey = pairwise_tukeyhsd(endog=v_concat, groups=group_names, alpha=alpha)
        print(tukey)

if __name__=='__main__':
    # computational efficiency
    ram_usage()
    estimate_gflops()
    peak_vram()
    explanation_latency()

    # Main AUC metrics
    average_AUC_scores()
    average_confidence_scores()
    auc_by_classes()

    # statistical significance
    statistical_significance()
