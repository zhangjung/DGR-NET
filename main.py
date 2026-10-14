"""Command-line entry point for training the dual-encoder RGB + depth deepfake detector."""
import argparse

from utils import logger, check_folder_structure
from trainer import run_dual_encoder_fusion_training


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dual Encoder (EfficientNet-B2) RGB + Depth Deepfake Detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths (required)
    parser.add_argument('--train-rgb', type=str, required=True,
                        help='Root folder of training RGB images (must contain 0_real/ and 1_fake/ subfolders)')
    parser.add_argument('--train-depth', type=str, required=True,
                        help='Root folder of training depth images (mirror structure of --train-rgb)')
    parser.add_argument('--test-rgb', type=str, required=True,
                        help='Root folder of test RGB images')
    parser.add_argument('--test-depth', type=str, required=True,
                        help='Root folder of test depth images')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory where the best model and visualization outputs are saved')

    # Training hyper-parameters
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for AdamW optimizer')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (auto-detected from GPU memory if not set)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='DataLoader worker processes (auto-detected if not set)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Multi-task loss weights
    parser.add_argument('--cls-weight', type=float, default=1.0,
                        help='Weight of the classification (focal) loss term')
    parser.add_argument('--recon-weight', type=float, default=0.1,
                        help='Weight of the depth reconstruction (MSE) loss term')

    # Utility modes
    parser.add_argument('--check-only', action='store_true',
                        help='Only validate the folder structure (no training)')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.check_only:
        logger.info("Folder structure check mode")
        logger.info("=" * 70)
        logger.info("Checking training data folder structure:")
        check_folder_structure(args.train_rgb, args.train_depth)
        logger.info("\n" + "=" * 70)
        logger.info("Checking test data folder structure:")
        check_folder_structure(args.test_rgb, args.test_depth)
        logger.info("\nFolder structure check completed!")
        return

    logger.info("Starting training: dual-encoder simplified RGB-SV-depth classifier")
    logger.info("=" * 70)
    logger.info(f"🚀 Core feature: dual EfficientNet-B2 encoders")
    logger.info(f"💡 Fusion strategy: stage-wise addition (stages 1-7)")
    logger.info(f"🔗 Skip connections: direct (no compression, full feature channels)")
    logger.info(f"📊 Classification: final-stage concatenation (1408+1408=2816)")
    logger.info(f"⚖️ Loss weights - classification: {args.cls_weight}, reconstruction: {args.recon_weight}")
    logger.info(f"🎨 Visualizations: 5 real + 5 fake reconstruction comparison + statistics")
    logger.info("=" * 70)

    model, best_model_path, best_accuracy = run_dual_encoder_fusion_training(
        train_rgb_folder=args.train_rgb,
        train_depth_folder=args.train_depth,
        test_rgb_folder=args.test_rgb,
        test_depth_folder=args.test_depth,
        model_output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        random_seed=args.seed,
        num_workers=args.num_workers,
        classification_weight=args.cls_weight,
        reconstruction_weight=args.recon_weight,
    )

    logger.info("\n" + "=" * 70)
    logger.info("🎉 Dual-encoder simplified training complete!")
    logger.info("=" * 70)
    logger.info(f"📈 Final accuracy: {best_accuracy:.4f}")
    logger.info(f"💾 Best model path: {best_model_path}")

    logger.info("\n🎯 Architecture highlights:")
    logger.info("   🔄 Dual EfficientNet-B2: symmetric encoder design")
    logger.info("   ➕ Stage-wise addition: feature fusion for stages 1-7")
    logger.info("   🔗 Direct skip: full channel features preserved (no compression)")
    logger.info("   📎 Final concatenation: preserves modality independence")
    logger.info("   🎨 Depth reconstruction: auxiliary task improving feature quality")
    logger.info("   📊 Visualizations: dedicated per-class comparison analyses")


if __name__ == '__main__':
    main()
