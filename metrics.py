from sklearn.metrics import auc
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from scipy import stats
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from torchvision.datasets import CIFAR10
from utils import load_data_samples
from explanations import XAI_METHODS, NUMBER_OF_ITERATIONS
from perturbation_pipeline import PERTURBATION_RATIOS

def plot_avg_confidence_scores(scores, ratios=PERTURBATION_RATIOS, **config):
    # Parse plotting configuration
    mode = config.get('mode', "Deletion")
    save_path = config.get('save_path', None)
    save_only = config.get('save_only', False)

    fig, ax = plt.subplots(figsize=(8,5))
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

    plt.title(f"Feature {mode}", fontsize=18)
    plt.xlabel(f"Perturbation ratio (%)", fontsize=16)
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

def average_AUC_scores(perturbation_scores_pth="./perturbation_scores", methods=list(XAI_METHODS.keys())):
    # Compute feature perturbation scores for each XAI method
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

def average_confidence_scores(perturbation_scores_pth="./perturbation_scores", methods=list(XAI_METHODS.keys())):
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
    plot_avg_confidence_scores(avg_del_confidence, mode='Deletion')
    plot_avg_confidence_scores(avg_ins_confidence, mode='Insertion')

def explanation_latency(explanations_path="./explanations",
                        methods=list(XAI_METHODS.keys()),
                        correct_indices_pth="./ckpts/correct_preds_indices.npy"):
    """ TODO: add description. """
    print("\n--- Average Explanation Latency ---")
    correct_pred_idx = np.load(correct_indices_pth)
    N = len(correct_pred_idx)
    for m in methods:
        print(f"\n{m}:")
        durations = [np.load(f"{explanations_path}/{m}/duration_{i}.npy") for i in range(1, NUMBER_OF_ITERATIONS+1)]
        avg_duration_per_sample = np.mean(durations) / N
        print(f"Latency = {1e3 * avg_duration_per_sample:.2f} ms per image sample")
        print(f"Throughput = {1/avg_duration_per_sample:.2f} images/sec")

def plot_metrics_by_class(_metrics, _class_labels, _mode='Deletion'):
    """ Helper function to plot Average Deletion/Insertion AUC by class label and for each XAI method. """
    # Bar Plot of the metrics
    fig, ax = plt.subplots(figsize=(15, 4))
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
    plt.title(f'Metrics by class (Feature {_mode})')
    plt.xlabel('Class')
    plt.ylabel(f'AUC')
    plt.legend(loc='upper left')
    plt.show()

def correct_preds_barplot(_correct_preds, _class_labels):
    class_counts = np.unique(_correct_preds, return_counts=True)[1]
    print('Correct Prediction Counts:')
    for c, count in zip(_class_labels, class_counts):
        print(f"{c}: {class_counts}")
    fig, ax = plt.subplots()
    x = np.arange(1, len(_class_labels)+1)
    ax.bar(x, height=class_counts)
    ax.set_xticks(x)
    ax.set_xticklabels(_class_labels)
    plt.xticks(rotation=45)
    plt.title('Correctly Predicted Class Counts')
    plt.xlabel('Class')
    plt.ylabel(f'# correct')
    plt.show()

def auc_by_classes(perturbation_scores_pth="./perturbation_scores", methods=list(XAI_METHODS.keys())):
    """ TODO: add description. """
    print("\n--- Deletion/Insertion AUC by Classes ---")

    # Load the dataset class names and the correctly predicted test samples
    test_dataset = CIFAR10(root='data', train=False)
    class_labels = test_dataset.classes
    correct_samples = load_data_samples()
    y_correct = [_[1] for _ in correct_samples.dataset]

    # DELETEME - sanity check
    # ground_truth = [_[1] for _ in test_dataset]
    # correct_preds_barplot(ground_truth, class_labels)

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


# def get_metrics_by_class(_embeds1, _embeds2, _all_labels, _k=5, _debug=False):
#     """ Helper function to compute mAP@k, recall@k, and precision@k for each 
#         of the target classes.
#     """
#     labels = [int(l.item()) for l in _all_labels]
#     metrics_by_class = {c: {} for c in tgt_classes}
#     for i, c in enumerate(tgt_classes):
#         class_embeds1 = [aud_emb.unsqueeze(0) for aud_emb, label in zip(_embeds1, _all_labels) if int(label.item()) == i]
#         class_embeds1 = torch.cat(class_embeds1, dim=0)
        
#         # construct relevance ranks
#         relevance = torch.zeros(_all_labels.size(0), device=_all_labels.device)
#         mask = (_all_labels == i)
#         idx = torch.nonzero(mask, as_tuple=True)[0].tolist()
#         relevance[idx] = 1
#         relevance = repeat(relevance.unsqueeze(0), '1 d -> b d', b=len(class_embeds1))
    
#         # metrics by class
#         mAP = map_at_k(class_embeds1, _embeds2, relevance, k=_k)
#         recall = recall_at_k(class_embeds1, _embeds2, relevance, k=_k)
#         precision = precision_at_k(class_embeds1, _embeds2, relevance, k=_k)
#         scores = {f'mAP@{_k}': mAP, f'recall@{_k}': recall, f'precision@{_k}': precision}
#         metrics_by_class[c] = scores

#         if _debug:
#             print(f'{c}: mAP@{_k}={mAP}, recall@{_k}={recall}, precision@{_k}={precision}')

#     return metrics_by_class

def statistical_significance(perturbation_scores_pth="./perturbation_scores", methods=list(XAI_METHODS.keys()), alpha=0.05):
    """
    TODO: add description

    Args:
        test_results_path (str): path to the directory containing the test results
        agent_types (list): list of agent types (e.g., ['SARSA', 'QLearning', 'SARSA_Lambda', 'DQN'])
        alpha (float): significance level for the statistical tests
    Returns:
        None
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
    # average_AUC_scores()
    # average_confidence_scores()
    # explanation_latency()
    auc_by_classes()
    # statistical_significance()
