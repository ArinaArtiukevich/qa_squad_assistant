import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd


def parse_training_log(file_path):
    metrics_data = defaultdict(list)
    current_metric = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            metric_match = re.match(r'^Metric: (.+)$', line)
            if metric_match:
                current_metric = metric_match.group(1)
                continue

            if current_metric and line.startswith('Step:'):
                step_match = re.match(r'Step: (\d+), Value: ([\d\.eE\-]+), Time: ([\d\.]+)', line)
                if step_match:
                    step = int(step_match.group(1))
                    value = float(step_match.group(2))
                    time = float(step_match.group(3))

                    metrics_data[current_metric].append({
                        'step': step,
                        'value': value,
                        'time': time
                    })

    return metrics_data


def load_and_combine_data(log_file_path, csv_file_path):
    log_data = parse_training_log(log_file_path)
    csv_data = pd.read_csv(csv_file_path)

    log_dfs = {}
    for metric, data in log_data.items():
        log_dfs[metric] = pd.DataFrame(data)

    combined_data = csv_data.copy()

    for metric, df in log_dfs.items():
        if metric.startswith('train/'):
            combined_data = combined_data.merge(
                df[['step', 'value']].rename(columns={'value': metric}),
                on='step', how='left'
            )

    return combined_data, log_dfs, csv_data


def plot_eval_metrics_over_time(combined_data, output_dir):
    eval_data = combined_data[combined_data['eval_exact'].notna()]
    metrics = ['eval_exact', 'eval_f1']
    colors = ['blue', 'green']

    plt.figure(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        plt.plot(eval_data['step'], eval_data[metric],
                 label=metric.replace('eval_', '').replace('_', ' ').title(),
                 color=colors[i], linewidth=2, marker='o')

    plt.xlabel('Evaluation Steps')
    plt.ylabel('Score (%)')
    plt.title('Evaluation Metrics Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eval_metrics_over_time.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_train_vs_valid_loss(combined_data, output_dir):
    train_loss = combined_data[combined_data['eval_exact'].isna()]['loss']
    train_steps = combined_data[combined_data['eval_exact'].isna()]['step']
    eval_loss = combined_data[combined_data['eval_exact'].notna()]['eval_loss']
    eval_steps = combined_data[combined_data['eval_exact'].notna()]['step']

    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_loss, 'b-', label='Training Loss', alpha=0.7)
    plt.plot(eval_steps, eval_loss, 'r-', label='Validation Loss', linewidth=2, marker='s')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'train_vs_valid_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_precision_recall_over_time(combined_data, output_dir):
    eval_data = combined_data[combined_data['eval_exact'].notna()]

    plt.figure(figsize=(10, 6))
    plt.plot(eval_data['step'], eval_data['eval_precision_answerable'],
             'c-', label='Precision', linewidth=2, marker='^')
    plt.plot(eval_data['step'], eval_data['eval_recall_answerable'],
             'm-', label='Recall', linewidth=2, marker='v')
    plt.xlabel('Evaluation Steps')
    plt.ylabel('Score')
    plt.title('Precision and Recall Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_over_time.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_metrics(combined_data, output_dir):
    eval_data = combined_data[combined_data['eval_exact'].notna()]
    confusion_metrics = ['eval_confusion_tp', 'eval_confusion_tn', 'eval_confusion_fp', 'eval_confusion_fn']
    labels = ['True Positives', 'True Negatives', 'False Positives', 'False Negatives']

    plt.figure(figsize=(12, 7))

    for i, metric in enumerate(confusion_metrics):
        plt.plot(eval_data['step'], eval_data[metric],
                 label=labels[i], linewidth=2)

    plt.xlabel('Evaluation Steps')
    plt.ylabel('Count')
    plt.title('Confusion Matrix Metrics Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_final_model_performance(combined_data, output_dir):
    eval_data = combined_data[combined_data['eval_exact'].notna()]

    if not eval_data.empty:
        plt.figure(figsize=(8, 6))
        metrics = ['eval_exact', 'eval_f1']
        values = [eval_data[metric].iloc[-1] for metric in metrics]
        labels = ['Exact Match', 'F1 Score']

        bars = plt.bar(labels, values, color=['skyblue', 'lightgreen'])
        plt.ylabel('Score (%)')
        plt.title('Final Model Performance')
        plt.ylim(0, 100)

        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{value:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'final_model_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()


def plot_training_metrics(combined_data, output_dir):
    train_data = combined_data[combined_data['eval_exact'].isna()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(train_data['step'], train_data['loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True, alpha=0.3)

    if 'train/grad_norm' in combined_data.columns:
        ax2.plot(train_data['step'], train_data['train/grad_norm'], 'g-', linewidth=2)
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Gradient Norm')
        ax2.set_title('Gradient Norm Over Time')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_all_plots(combined_data, output_dir):
    plot_eval_metrics_over_time(combined_data, output_dir)
    plot_train_vs_valid_loss(combined_data, output_dir)
    plot_precision_recall_over_time(combined_data, output_dir)
    plot_confusion_metrics(combined_data, output_dir)
    plot_final_model_performance(combined_data, output_dir)
    plot_training_metrics(combined_data, output_dir)


def print_combined_analysis(combined_data):
    print("=== COMBINED ANALYSIS SUMMARY ===")
    print(f"Total training steps: {len(combined_data[combined_data['eval_exact'].isna()])}")
    print(f"Total evaluation points: {len(combined_data[combined_data['eval_exact'].notna()])}")
    print(f"Final epoch: {combined_data['epoch'].max():.2f}")

    eval_data = combined_data[combined_data['eval_exact'].notna()]
    if not eval_data.empty:
        print("\n=== EVALUATION PERFORMANCE ===")

        best_exact = eval_data.loc[eval_data['eval_exact'].idxmax()]
        best_f1 = eval_data.loc[eval_data['eval_f1'].idxmax()]

        print(
            f"Best Exact Match: {best_exact['eval_exact']:.2f}% at step {best_exact['step']} (epoch {best_exact['epoch']:.2f})")
        print(f"Best F1 Score: {best_f1['eval_f1']:.2f}% at step {best_f1['step']} (epoch {best_f1['epoch']:.2f})")

        final_metrics = eval_data.iloc[-1]
        print(f"\nFinal Performance (Step {final_metrics['step']}, Epoch {final_metrics['epoch']:.2f}):")
        print(f"  Exact Match: {final_metrics['eval_exact']:.2f}%")
        print(f"  F1 Score: {final_metrics['eval_f1']:.2f}%")
        print(f"  No-Answer Accuracy: {final_metrics['eval_no_answer_accuracy']:.3f}")
        print(f"  Validation Loss: {final_metrics['eval_loss']:.4f}")


def print_confusion_matrix_analysis(combined_data):
    eval_data = combined_data[combined_data['eval_exact'].notna()]

    if eval_data.empty:
        print("No evaluation data found!")
        return

    print("=" * 60)
    print("CONFUSION MATRIX ANALYSIS")
    print("=" * 60)

    for idx, row in eval_data.iterrows():
        print(f"\n--- Evaluation at Step {int(row['step'])} (Epoch {row['epoch']:.2f}) ---")

        tp = int(row['eval_confusion_tp'])
        tn = int(row['eval_confusion_tn'])
        fp = int(row['eval_confusion_fp'])
        fn = int(row['eval_confusion_fn'])

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"True Positives (TP):  {tp:>6}")
        print(f"True Negatives (TN):  {tn:>6}")
        print(f"False Positives (FP): {fp:>6}")
        print(f"False Negatives (FN): {fn:>6}")
        print(f"Total Samples:        {total:>6}")
        print(f"Accuracy:             {accuracy:.4f}")
        print(f"Precision:            {precision:.4f}")
        print(f"Recall:               {recall:.4f}")
        print(f"F1 Score:             {f1:.4f}")

        print("\nConfusion Matrix:")
        print(f"               Predicted")
        print(f"               Positive   Negative")
        print(f"Actual Positive  {tp:>4} TP     {fn:>4} FN")
        print(f"Actual Negative  {fp:>4} FP     {tn:>4} TN")

    final_eval = eval_data.iloc[-1]
    tp_final = int(final_eval['eval_confusion_tp'])
    tn_final = int(final_eval['eval_confusion_tn'])
    fp_final = int(final_eval['eval_confusion_fp'])
    fn_final = int(final_eval['eval_confusion_fn'])

    print("\n" + "=" * 60)
    print("FINAL CONFUSION MATRIX SUMMARY")
    print("=" * 60)
    print(f"Step: {int(final_eval['step'])}, Epoch: {final_eval['epoch']:.2f}")
    print(f"TP: {tp_final}, TN: {tn_final}, FP: {fp_final}, FN: {fn_final}")

    total_final = tp_final + tn_final + fp_final + fn_final
    print(f"Total samples: {total_final}")
    print(f"Accuracy: {(tp_final + tn_final) / total_final:.4f}")


if __name__ == "__main__":
    output_dir = "/Users/arynaartsiukevich/PycharmProjects/qa_sqaud_assistant/src/input/results/"

    log_file_path = os.path.join(output_dir, 'log.txt')
    csv_file_path = os.path.join(output_dir, 'training_logs.csv')

    os.makedirs(output_dir, exist_ok=True)

    combined_data, log_dfs, csv_data = load_and_combine_data(log_file_path, csv_file_path)
    create_all_plots(combined_data, output_dir)
    print_combined_analysis(combined_data)
    print_confusion_matrix_analysis(combined_data)

    print(f"\nPlots saved to: {output_dir}")
