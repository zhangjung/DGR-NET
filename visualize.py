"""All visualization functions: training metrics, confusion matrix, t-SNE,
depth reconstruction samples, per-class reconstructions, and reconstruction statistics."""
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from tqdm import tqdm

from utils import logger


# =============================================================================
# Per-class reconstruction visualization (5 each for real/fake)
# =============================================================================
def visualize_reconstruction_by_class(model, dataloader, device, output_dir, samples_per_class=5):
    """
    Visualize depth reconstruction results grouped by class.
    Args:
        model: trained model
        dataloader: data loader
        device: torch device
        output_dir: output directory
        samples_per_class: number of samples per class (default 5)
    """
    logger.info(f"Generating per-class reconstruction visualizations ({samples_per_class} each for real/fake)...")

    model.eval()

    # Collect samples per class
    real_samples = []  # 0_real
    fake_samples = []  # 1_fake

    # Collect samples
    with torch.no_grad():
        for rgb_inputs, depth_inputs, labels in dataloader:
            rgb_inputs = rgb_inputs.to(device)
            depth_inputs = depth_inputs.to(device)
            labels = labels.to(device)

            outputs = model(rgb_inputs, depth_inputs)

            # Collect samples by class
            for i in range(rgb_inputs.size(0)):
                label = labels[i].item()

                sample_data = {
                    'rgb': rgb_inputs[i],
                    'depth': depth_inputs[i],
                    'label': label,
                    'rgb_sv_fused': outputs['rgb_sv_fused'][i],
                    'reconstructed_depth': outputs['reconstructed_depth'][i],
                    'classification': outputs['classification'][i]
                }

                if label == 0 and len(real_samples) < samples_per_class:
                    real_samples.append(sample_data)
                elif label == 1 and len(fake_samples) < samples_per_class:
                    fake_samples.append(sample_data)

                # Stop early once both classes have enough samples
                if len(real_samples) >= samples_per_class and len(fake_samples) >= samples_per_class:
                    break

            if len(real_samples) >= samples_per_class and len(fake_samples) >= samples_per_class:
                break

    # Check whether we have enough samples
    if len(real_samples) < samples_per_class:
        logger.warning(f"Not enough real samples: found {len(real_samples)}, need {samples_per_class}")
    if len(fake_samples) < samples_per_class:
        logger.warning(f"Not enough fake samples: found {len(fake_samples)}, need {samples_per_class}")

    # Generate visualizations
    # Option 1: combined figure with all samples
    _create_combined_visualization(real_samples, fake_samples, output_dir, samples_per_class)

    # Option 2: separate per-class figures
    _create_class_wise_visualization(real_samples, fake_samples, output_dir, samples_per_class)

    logger.info(f"Per-class reconstruction visualizations saved to: {output_dir}")


def _create_combined_visualization(real_samples, fake_samples, output_dir, samples_per_class):
    """Build a combined visualization (both real and fake samples on a single figure)."""

    # Layout
    n_cols = 5
    n_rows = len(real_samples) + len(fake_samples) + 2

    fig = plt.figure(figsize=(n_cols * 4, n_rows * 3))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)

    current_row = 0

    # ===== Real samples section =====
    title_ax = fig.add_subplot(gs[current_row, :])
    title_ax.text(0.5, 0.5, '🟢 REAL SAMPLES (Ground Truth)',
                  ha='center', va='center', fontsize=20, fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    title_ax.axis('off')
    current_row += 1

    # Draw real samples
    for idx, sample in enumerate(real_samples):
        _plot_sample_row(fig, gs, current_row, sample, idx + 1, 'Real')
        current_row += 1

    # ===== Fake samples section =====
    title_ax = fig.add_subplot(gs[current_row, :])
    title_ax.text(0.5, 0.5, '🔴 FAKE SAMPLES (AI Generated)',
                  ha='center', va='center', fontsize=20, fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    title_ax.axis('off')
    current_row += 1

    # Draw fake samples
    for idx, sample in enumerate(fake_samples):
        _plot_sample_row(fig, gs, current_row, sample, idx + 1, 'Fake')
        current_row += 1

    # Save combined figure
    combined_path = os.path.join(output_dir, 'reconstruction_combined_real_fake.png')
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Combined visualization saved: {combined_path}")


def _create_class_wise_visualization(real_samples, fake_samples, output_dir, samples_per_class):
    """Build per-class visualizations (separate figures for real and fake)."""

    # ===== Real samples figure =====
    if real_samples:
        n_cols = 5
        n_rows = len(real_samples)

        fig = plt.figure(figsize=(n_cols * 4, n_rows * 3))
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)

        for idx, sample in enumerate(real_samples):
            _plot_sample_row(fig, gs, idx, sample, idx + 1, 'Real')

        fig.suptitle('🟢 REAL SAMPLES - Depth Reconstruction Results',
                     fontsize=18, fontweight='bold', y=0.995)

        real_path = os.path.join(output_dir, 'reconstruction_real_samples.png')
        plt.savefig(real_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Real samples visualization saved: {real_path}")

    # ===== Fake samples figure =====
    if fake_samples:
        n_cols = 5
        n_rows = len(fake_samples)

        fig = plt.figure(figsize=(n_cols * 4, n_rows * 3))
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)

        for idx, sample in enumerate(fake_samples):
            _plot_sample_row(fig, gs, idx, sample, idx + 1, 'Fake')

        fig.suptitle('🔴 FAKE SAMPLES - Depth Reconstruction Results',
                     fontsize=18, fontweight='bold', y=0.995)

        fake_path = os.path.join(output_dir, 'reconstruction_fake_samples.png')
        plt.savefig(fake_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Fake samples visualization saved: {fake_path}")


def _plot_sample_row(fig, gs, row, sample, sample_idx, class_name):
    """
    Plot a single sample as one row (5 columns).
    Args:
        fig: matplotlib figure
        gs: GridSpec
        row: current row index
        sample: sample data dict
        sample_idx: sample number
        class_name: class name ('Real' or 'Fake')
    """

    # Extract data
    rgb_img = sample['rgb'].cpu().permute(1, 2, 0).numpy()
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)

    rgb_sv_fused = sample['rgb_sv_fused'].cpu().permute(1, 2, 0).numpy()
    rgb_sv_fused = (rgb_sv_fused - rgb_sv_fused.min()) / (rgb_sv_fused.max() - rgb_sv_fused.min() + 1e-8)

    original_depth = sample['depth'].cpu().squeeze().numpy()
    reconstructed_depth = sample['reconstructed_depth'].cpu().squeeze().numpy()
    depth_error = np.abs(original_depth - reconstructed_depth)

    pred_prob = torch.sigmoid(sample['classification']).item()
    pred_class = 'Fake' if pred_prob > 0.5 else 'Real'

    # Compute reconstruction quality metrics
    mse = np.mean(depth_error ** 2)
    mae = np.mean(depth_error)

    # Column 1: Original RGB
    ax1 = fig.add_subplot(gs[row, 0])
    ax1.imshow(rgb_img)
    ax1.set_title(f'{class_name} #{sample_idx}\nOriginal RGB', fontsize=10, fontweight='bold')
    ax1.axis('off')

    # Column 2: RGB-SV fused
    ax2 = fig.add_subplot(gs[row, 1])
    ax2.imshow(rgb_sv_fused)
    ax2.set_title('RGB-SV Fused\n(Encoder Input)', fontsize=10)
    ax2.axis('off')

    # Column 3: Original depth
    ax3 = fig.add_subplot(gs[row, 2])
    im3 = ax3.imshow(original_depth, cmap='plasma')
    ax3.set_title('Original Depth\n(Ground Truth)', fontsize=10)
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # Column 4: Reconstructed depth
    ax4 = fig.add_subplot(gs[row, 3])
    im4 = ax4.imshow(reconstructed_depth, cmap='plasma')
    ax4.set_title('Reconstructed Depth\n(Model Output)', fontsize=10)
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # Column 5: Error map + statistics
    ax5 = fig.add_subplot(gs[row, 4])
    im5 = ax5.imshow(depth_error, cmap='hot')
    ax5.set_title('Reconstruction Error', fontsize=10)
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    # Add text info below the error map
    info_text = f'Prediction: {pred_class} ({pred_prob:.3f})\n'
    info_text += f'MSE: {mse:.4f}\n'
    info_text += f'MAE: {mae:.4f}\n'

    # Check whether the prediction is correct
    is_correct = (pred_class == class_name)
    info_text += f'Status: {"✅ Correct" if is_correct else "❌ Wrong"}'

    # Add text below the figure
    ax5.text(0.5, -0.15, info_text,
             transform=ax5.transAxes,
             ha='center', va='top',
             fontsize=8,
             bbox=dict(boxstyle='round',
                       facecolor='lightgreen' if is_correct else 'lightcoral',
                       alpha=0.6))


def visualize_reconstruction_statistics(model, dataloader, device, output_dir):
    """
    Generate a statistical analysis figure of reconstruction quality.
    Compare the reconstruction-error distributions between real and fake samples.
    """
    logger.info("Generating reconstruction-quality statistics...")

    model.eval()

    real_mse_list = []
    fake_mse_list = []
    real_mae_list = []
    fake_mae_list = []

    with torch.no_grad():
        for rgb_inputs, depth_inputs, labels in tqdm(dataloader, desc="Computing reconstruction error"):
            rgb_inputs = rgb_inputs.to(device)
            depth_inputs = depth_inputs.to(device)
            labels = labels.to(device)

            outputs = model(rgb_inputs, depth_inputs)

            # Compute per-sample reconstruction error
            for i in range(rgb_inputs.size(0)):
                original = depth_inputs[i].cpu().numpy()
                reconstructed = outputs['reconstructed_depth'][i].cpu().numpy()

                mse = np.mean((original - reconstructed) ** 2)
                mae = np.mean(np.abs(original - reconstructed))

                if labels[i].item() == 0:  # Real
                    real_mse_list.append(mse)
                    real_mae_list.append(mae)
                else:  # Fake
                    fake_mse_list.append(mse)
                    fake_mae_list.append(mae)

    # Build the statistics figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # MSE distribution histogram
    axes[0, 0].hist(real_mse_list, bins=50, alpha=0.6, color='green', label='Real', edgecolor='black')
    axes[0, 0].hist(fake_mse_list, bins=50, alpha=0.6, color='red', label='Fake', edgecolor='black')
    axes[0, 0].set_xlabel('MSE')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('MSE Distribution: Real vs Fake')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # MAE distribution histogram
    axes[0, 1].hist(real_mae_list, bins=50, alpha=0.6, color='green', label='Real', edgecolor='black')
    axes[0, 1].hist(fake_mae_list, bins=50, alpha=0.6, color='red', label='Fake', edgecolor='black')
    axes[0, 1].set_xlabel('MAE')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('MAE Distribution: Real vs Fake')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # MSE boxplot comparison
    axes[1, 0].boxplot([real_mse_list, fake_mse_list],
                       labels=['Real', 'Fake'],
                       patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.6),
                       medianprops=dict(color='red', linewidth=2))
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('MSE Comparison: Real vs Fake')
    axes[1, 0].grid(True, alpha=0.3)

    # MAE boxplot comparison
    axes[1, 1].boxplot([real_mae_list, fake_mae_list],
                       labels=['Real', 'Fake'],
                       patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.6),
                       medianprops=dict(color='red', linewidth=2))
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].set_title('MAE Comparison: Real vs Fake')
    axes[1, 1].grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f'Statistics Summary:\n\n'
    stats_text += f'Real Samples:\n'
    stats_text += f'  MSE: {np.mean(real_mse_list):.4f} ± {np.std(real_mse_list):.4f}\n'
    stats_text += f'  MAE: {np.mean(real_mae_list):.4f} ± {np.std(real_mae_list):.4f}\n\n'
    stats_text += f'Fake Samples:\n'
    stats_text += f'  MSE: {np.mean(fake_mse_list):.4f} ± {np.std(fake_mse_list):.4f}\n'
    stats_text += f'  MAE: {np.mean(fake_mae_list):.4f} ± {np.std(fake_mae_list):.4f}'

    fig.text(0.5, 0.02, stats_text, ha='center', va='bottom', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    stats_path = os.path.join(output_dir, 'reconstruction_statistics.png')
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Reconstruction statistics saved: {stats_path}")


# =============================================================================
# Other visualization functions
# =============================================================================
def visualize_dual_encoder_reconstruction(model, dataloader, device, output_dir, num_samples=4):
    """Visualize dual-encoder depth reconstruction results."""
    logger.info("Visualizing dual encoder depth reconstruction results...")

    model.eval()
    sample_count = 0

    with torch.no_grad():
        for rgb_inputs, depth_inputs, labels in dataloader:
            if sample_count >= num_samples:
                break

            rgb_inputs = rgb_inputs.to(device)
            depth_inputs = depth_inputs.to(device)

            outputs = model(rgb_inputs, depth_inputs)

            batch_size = rgb_inputs.size(0)
            for i in range(min(batch_size, num_samples - sample_count)):
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))

                # Original RGB
                rgb_img = rgb_inputs[i].cpu().permute(1, 2, 0).numpy()
                rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
                axes[0, 0].imshow(rgb_img)
                axes[0, 0].set_title('Original RGB')
                axes[0, 0].axis('off')

                # RGB-SV fused
                fused_img = outputs['rgb_sv_fused'][i].cpu().permute(1, 2, 0).numpy()
                fused_img = (fused_img - fused_img.min()) / (fused_img.max() - fused_img.min())
                axes[0, 1].imshow(fused_img)
                axes[0, 1].set_title('RGB-SV Fused')
                axes[0, 1].axis('off')

                # Original depth
                original_depth = depth_inputs[i].cpu().squeeze().numpy()
                im1 = axes[0, 2].imshow(original_depth, cmap='plasma')
                axes[0, 2].set_title('Original Depth (Ground Truth)')
                axes[0, 2].axis('off')
                plt.colorbar(im1, ax=axes[0, 2], fraction=0.046)

                # Reconstructed depth
                recon_depth = outputs['reconstructed_depth'][i].cpu().squeeze().numpy()
                im2 = axes[1, 0].imshow(recon_depth, cmap='plasma')
                axes[1, 0].set_title('Reconstructed Depth (Simplified)')
                axes[1, 0].axis('off')
                plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

                # Depth reconstruction error
                depth_error = np.abs(original_depth - recon_depth)
                im3 = axes[1, 1].imshow(depth_error, cmap='hot')
                axes[1, 1].set_title('Depth Reconstruction Error')
                axes[1, 1].axis('off')
                plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

                # Prediction summary
                pred_prob = torch.sigmoid(outputs['classification'][i]).item()
                label_text = f'Label: {"Fake" if labels[i] == 1 else "Real"}\n'
                label_text += f'Prediction: {pred_prob:.3f}\n'
                label_text += f'Predicted: {"Fake" if pred_prob > 0.5 else "Real"}\n\n'

                mse = np.mean(depth_error ** 2)
                mae = np.mean(depth_error)
                label_text += f'Depth MSE: {mse:.4f}\n'
                label_text += f'Depth MAE: {mae:.4f}\n\n'
                label_text += 'Architecture:\n'
                label_text += '- Dual EfficientNet B2\n'
                label_text += '- Layer-wise Addition\n'
                label_text += '- Direct Skip Connections\n'
                label_text += '- Last Layer Concatenation'

                axes[1, 2].text(0.5, 0.5, label_text,
                                horizontalalignment='center',
                                verticalalignment='center',
                                transform=axes[1, 2].transAxes,
                                fontsize=9,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                axes[1, 2].set_title('Results & Architecture')
                axes[1, 2].axis('off')

                plt.suptitle(f'Dual Encoder Simplified - Sample {sample_count + 1}')
                plt.tight_layout()

                viz_path = os.path.join(output_dir,
                                        f'dual_encoder_reconstruction_sample_{sample_count + 1}.png')
                plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close()

                sample_count += 1
                if sample_count >= num_samples:
                    break

    logger.info(f"Dual encoder reconstruction visualization saved to: {output_dir}")


def plot_training_metrics(train_losses, test_losses, train_accs, test_accs, output_dir,
                          model_info="Dual Encoder Simplified"):
    """Plot training metrics."""
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, train_losses, 'b-o', label='Training Loss', markersize=4)
    axes[0].plot(epochs, test_losses, 'r-o', label='Testing Loss', markersize=4)
    axes[0].set_title(f'Loss over Epochs ({model_info})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, train_accs, 'b-o', label='Training Accuracy', markersize=4)
    axes[1].plot(epochs, test_accs, 'r-o', label='Testing Accuracy', markersize=4)
    axes[1].set_title(f'Accuracy over Epochs ({model_info})')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    metrics_path = os.path.join(output_dir, f'training_metrics_{model_info.lower().replace(" ", "_")}.png')
    plt.savefig(metrics_path)
    plt.close()
    logger.info(f"Training metrics saved to {metrics_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir, model_info="Dual Encoder"):
    """Plot the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.title(f'Confusion Matrix ({model_info})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_path = os.path.join(output_dir, f'confusion_matrix_{model_info.lower().replace(" ", "_")}.png')
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Confusion Matrix saved to {cm_path}")


def plot_tsne_efficient(model, dataloader, device, class_names, output_dir, max_samples=2000,
                        model_info="Dual Encoder"):
    """Efficient t-SNE visualization."""
    logger.info(f"Generating t-SNE plot for {model_info.lower()} features...")

    model.eval()
    all_features = []
    all_labels = []
    sample_count = 0

    with torch.no_grad():
        for rgb_inputs, depth_inputs, labels in dataloader:
            if sample_count >= max_samples:
                break

            rgb_inputs = rgb_inputs.to(device)
            depth_inputs = depth_inputs.to(device)
            features = model.get_features(rgb_inputs, depth_inputs)

            batch_size = features.size(0)
            remaining_samples = max_samples - sample_count
            actual_batch_size = min(batch_size, remaining_samples)

            all_features.append(features[:actual_batch_size].cpu().numpy())
            all_labels.extend(labels[:actual_batch_size].cpu().numpy())
            sample_count += actual_batch_size

            if sample_count >= max_samples:
                break

    if not all_features:
        logger.warning("No features extracted for t-SNE")
        return

    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features) // 4), n_iter=1000)
    tsne_results = tsne.fit_transform(all_features)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=all_labels,
        palette=sns.color_palette("hsv", len(class_names)),
        legend="full",
        alpha=0.8,
        s=10
    )
    plt.title(f't-SNE Feature Visualization ({model_info})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=class_names, title="Class")

    tsne_path = os.path.join(output_dir, f't_sne_plot_{model_info.lower().replace(" ", "_")}.png')
    plt.savefig(tsne_path)
    plt.close()
    logger.info(f"t-SNE plot saved to: {tsne_path}")
