"""Train/evaluate step functions, LR scheduler, and the main training pipeline."""
import os
import sys
import math
import gc

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

from utils import (logger, set_random_seeds, get_optimal_workers,
                   get_optimized_batch_size, check_folder_structure)
from dataset import HighPerformanceRGBDepthDataset, get_paired_transforms
from models import DualEncoderLayerFusionModel, UNetMultiTaskLoss
from visualize import (plot_training_metrics, plot_confusion_matrix,
                       visualize_dual_encoder_reconstruction,
                       visualize_reconstruction_by_class,
                       visualize_reconstruction_statistics,
                       plot_tsne_efficient)


def get_enhanced_scheduler(optimizer, num_epochs, warmup_epochs=3):
    """Enhanced learning-rate scheduler with linear warmup + cosine decay."""

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            cos_epoch = epoch - warmup_epochs
            cos_total = num_epochs - warmup_epochs
            return 0.5 * (1 + math.cos(math.pi * cos_epoch / cos_total))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Train / evaluate step functions
# =============================================================================
def train_dual_encoder_model(model, dataloader, criterion, optimizer, device, accumulation_steps=1):
    """Training step for one epoch."""
    model.train()
    running_loss = 0.0
    running_class_loss = 0.0
    running_recon_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training (Dual Encoder - Simplified)", unit="batch")
    for batch_idx, (rgb_inputs, depth_inputs, labels) in enumerate(pbar):
        rgb_inputs = rgb_inputs.to(device, non_blocking=True)
        depth_inputs = depth_inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(rgb_inputs, depth_inputs)
        total_loss, loss_dict = criterion(outputs, labels)

        loss = total_loss / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        batch_size = rgb_inputs.size(0)
        running_loss += loss_dict['total_loss'].item() * batch_size
        running_class_loss += loss_dict['classification_loss'].item() * batch_size
        running_recon_loss += loss_dict['reconstruction_loss'].item() * batch_size

        probs = torch.sigmoid(outputs['classification'])
        preds = (probs > 0.5).float()
        correct_predictions += (preds == labels.view(-1, 1)).sum().item()
        total_samples += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({
            'Total': f'{loss_dict["total_loss"].item():.4f}',
            'Class': f'{loss_dict["classification_loss"].item():.4f}',
            'Recon': f'{loss_dict["reconstruction_loss"].item():.4f}',
            'Acc': f'{correct_predictions / total_samples:.4f}'
        })

    avg_loss = running_loss / total_samples
    avg_class_loss = running_class_loss / total_samples
    avg_recon_loss = running_recon_loss / total_samples
    accuracy = correct_predictions / total_samples
    f1 = f1_score(all_labels, all_preds)

    return avg_loss, avg_class_loss, avg_recon_loss, accuracy, f1


def evaluate_dual_encoder_model(model, dataloader, criterion, device):
    """Evaluation step."""
    model.eval()
    running_loss = 0.0
    running_class_loss = 0.0
    running_recon_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating (Dual Encoder - Simplified)", unit="batch")
        for rgb_inputs, depth_inputs, labels in pbar:
            rgb_inputs = rgb_inputs.to(device, non_blocking=True)
            depth_inputs = depth_inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(rgb_inputs, depth_inputs)
            total_loss, loss_dict = criterion(outputs, labels)

            batch_size = rgb_inputs.size(0)
            running_loss += loss_dict['total_loss'].item() * batch_size
            running_class_loss += loss_dict['classification_loss'].item() * batch_size
            running_recon_loss += loss_dict['reconstruction_loss'].item() * batch_size

            probs = torch.sigmoid(outputs['classification'])
            preds = (probs > 0.5).float()
            correct_predictions += (preds == labels.view(-1, 1)).sum().item()
            total_samples += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'Total': f'{loss_dict["total_loss"].item():.4f}',
                'Class': f'{loss_dict["classification_loss"].item():.4f}',
                'Recon': f'{loss_dict["reconstruction_loss"].item():.4f}',
                'Acc': f'{correct_predictions / total_samples:.4f}'
            })

    avg_loss = running_loss / total_samples
    avg_class_loss = running_class_loss / total_samples
    avg_recon_loss = running_recon_loss / total_samples
    accuracy = correct_predictions / total_samples
    f1 = f1_score(all_labels, all_preds)

    return avg_loss, avg_class_loss, avg_recon_loss, accuracy, f1, np.array(all_preds), np.array(all_labels)


# =============================================================================
# Main training pipeline
# =============================================================================
def run_dual_encoder_fusion_training(train_rgb_folder, train_depth_folder,
                                     test_rgb_folder, test_depth_folder,
                                     model_output_dir, num_epochs=30, batch_size=None,
                                     learning_rate=0.0001, random_seed=42,
                                     num_workers=None,
                                     classification_weight=1.0, reconstruction_weight=0.5):
    """Run the dual-encoder stage-wise fusion training (simplified variant)."""

    # Validate folder structure
    logger.info("Checking training-data folder structure...")
    if not check_folder_structure(train_rgb_folder, train_depth_folder):
        sys.exit("Training-data folder structure check failed; please fix and retry.")

    logger.info("Checking test-data folder structure...")
    if not check_folder_structure(test_rgb_folder, test_depth_folder):
        sys.exit("Test-data folder structure check failed; please fix and retry.")

    logger.info("All folder structure checks passed; starting training...")

    set_random_seeds(random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if num_workers is None:
        num_workers = get_optimal_workers()
    if batch_size is None:
        batch_size = get_optimized_batch_size(device, model_size='b2')

    effective_batch_size = 32
    accumulation_steps = max(1, effective_batch_size // batch_size)

    logger.info(f"Dual-encoder simplified model configuration:")
    logger.info(f"  Actual batch size: {batch_size}")
    logger.info(f"  Gradient accumulation steps: {accumulation_steps}")
    logger.info(f"  Effective batch size: {batch_size * accumulation_steps}")
    logger.info(f"  Loss weights - classification: {classification_weight}, reconstruction: {reconstruction_weight}")

    rgb_train_transform = get_paired_transforms(is_training=True)
    rgb_test_transform = get_paired_transforms(is_training=False)

    try:
        train_dataset = HighPerformanceRGBDepthDataset(
            rgb_root_dir=train_rgb_folder,
            depth_root_dir=train_depth_folder,
            transform=rgb_train_transform
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else 2,
            drop_last=True
        )

        test_dataset = HighPerformanceRGBDepthDataset(
            rgb_root_dir=test_rgb_folder,
            depth_root_dir=test_depth_folder,
            transform=rgb_test_transform
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else 2
        )
    except RuntimeError as e:
        sys.exit(f"Data loading error: {e}")

    logger.info(f"Training set RGB-depth pairs: {len(train_dataset)}")
    logger.info(f"Test set RGB-depth pairs: {len(test_dataset)}")

    # Build the model
    model = DualEncoderLayerFusionModel(pretrained=True).to(device)

    # Loss function
    criterion = UNetMultiTaskLoss(
        classification_weight=classification_weight,
        reconstruction_weight=reconstruction_weight,
        use_focal_loss=True
    )

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Learning-rate scheduler
    scheduler = get_enhanced_scheduler(optimizer, num_epochs, warmup_epochs=3)

    os.makedirs(model_output_dir, exist_ok=True)

    best_accuracy = -1.0
    best_model_path = None
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    logger.info("Starting training of the dual-encoder simplified classifier...")
    for epoch in range(num_epochs):
        logger.info(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        logger.info(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        # Train
        train_loss, train_class_loss, train_recon_loss, train_acc, train_f1 = train_dual_encoder_model(
            model, train_loader, criterion, optimizer, device, accumulation_steps)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        logger.info(f"Epoch {epoch + 1} training complete:")
        logger.info(f"  Total loss: {train_loss:.4f} (classification: {train_class_loss:.4f}, reconstruction: {train_recon_loss:.4f})")
        logger.info(f"  Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")

        # Test
        if test_dataset:
            test_loss, test_class_loss, test_recon_loss, test_acc, test_f1, all_test_preds, all_test_labels = evaluate_dual_encoder_model(
                model, test_loader, criterion, device)

            test_losses.append(test_loss)
            test_accuracies.append(test_acc)

            logger.info(f"Epoch {epoch + 1} evaluation complete:")
            logger.info(f"  Total loss: {test_loss:.4f} (classification: {test_class_loss:.4f}, reconstruction: {test_recon_loss:.4f})")
            logger.info(f"  Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")

            # Save the best model
            if test_acc > best_accuracy:
                if best_model_path and os.path.exists(best_model_path):
                    try:
                        os.remove(best_model_path)
                    except Exception as e:
                        logger.error(f"Failed to delete the old model file: {e}")

                best_accuracy = test_acc
                accuracy_str = f"{test_acc:.4f}".replace('.', '_')
                best_model_path = os.path.join(model_output_dir,
                                               f'best_dual_encoder_simplified_acc_{accuracy_str}.pth')

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': test_acc,
                    'epoch': epoch,
                    'model_type': 'dual_encoder_simplified',
                    'classification_weight': classification_weight,
                    'reconstruction_weight': reconstruction_weight
                }, best_model_path)
                logger.info(f"!!! New best model found, test accuracy {test_acc:.4f}, saved to {best_model_path} !!!")

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info("\nDual-encoder simplified model training and evaluation complete.")
    if best_model_path:
        logger.info(f"Best model (accuracy: {best_accuracy:.4f}) saved to: {best_model_path}")

    # ========== Visualization ==========
    model.eval()
    if test_dataset:
        final_test_loss, final_test_class_loss, final_test_recon_loss, final_test_acc, final_test_f1, all_test_preds, all_test_labels = evaluate_dual_encoder_model(
            model, test_loader, criterion, device)

        logger.info(f"\nFinal test results:")
        logger.info(
            f"  Total loss: {final_test_loss:.4f} (classification: {final_test_class_loss:.4f}, reconstruction: {final_test_recon_loss:.4f})")
        logger.info(f"  Accuracy: {final_test_acc:.4f}, F1: {final_test_f1:.4f}")

        logger.info("\n" + "=" * 70)
        logger.info("Generating visualization outputs...")
        logger.info("=" * 70)

        model_info = "Dual Encoder Simplified"

        # 1. Training metrics
        logger.info("📊 1/6 Generating training-metrics plot...")
        plot_training_metrics(train_losses, test_losses, train_accuracies, test_accuracies, model_output_dir,
                              model_info)

        # 2. Confusion matrix
        logger.info("📊 2/6 Generating confusion matrix...")
        class_labels = ['Real', 'Fake']
        plot_confusion_matrix(all_test_labels, all_test_preds, class_labels, model_output_dir, model_info)

        # 3. Generic reconstruction visualization (4 samples)
        logger.info("📊 3/6 Generating dual-encoder depth reconstruction visualization...")
        visualize_dual_encoder_reconstruction(model, test_loader, device, model_output_dir, num_samples=4)

        # 4. Per-class reconstruction visualization (5 each for real/fake)
        logger.info("📊 4/6 Generating per-class reconstruction comparison (5 real + 5 fake)...")
        visualize_reconstruction_by_class(model, test_loader, device, model_output_dir, samples_per_class=5)

        # 5. Reconstruction-quality statistics
        logger.info("📊 5/6 Generating reconstruction-quality statistics...")
        visualize_reconstruction_statistics(model, test_loader, device, model_output_dir)

        # 6. t-SNE visualization
        logger.info("📊 6/6 Generating t-SNE visualization...")
        plot_tsne_efficient(model, test_loader, device, class_labels, model_output_dir, model_info=model_info)

    logger.info("\n" + "=" * 70)
    logger.info("✅ All visualizations generated!")
    logger.info("=" * 70)
    logger.info(f"\nGenerated visualization files:")
    logger.info(f"  1. training_metrics_*.png - training curves")
    logger.info(f"  2. confusion_matrix_*.png - confusion matrix")
    logger.info(f"  3. dual_encoder_reconstruction_sample_*.png - reconstruction examples (4)")
    logger.info(f"  4. reconstruction_combined_real_fake.png - combined real/fake figure (10 samples)")
    logger.info(f"  5. reconstruction_real_samples.png - real samples figure (5)")
    logger.info(f"  6. reconstruction_fake_samples.png - fake samples figure (5)")
    logger.info(f"  7. reconstruction_statistics.png - reconstruction error statistics")
    logger.info(f"  8. t_sne_plot_*.png - feature distribution visualization")

    return model, best_model_path, best_accuracy
