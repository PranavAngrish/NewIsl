import torch
import os
import json
import pickle
from dotenv import load_dotenv

from config import ISLConfig
from enhancedDataPreprocessor import EnhancedDataPreprocessor
from pytorchModelBuilder import ISLModelBuilder  # Assumes updated model builder uses EnhancedISLViTModel
from pytorchTrainer import EnhancedISLTrainer
from pytorchEvaluator import ISLEvaluator
from logger import setup_logger

# Improve MPS memory stability (for macOS users)
os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.5'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.7'

load_dotenv()

def main():
    logger = setup_logger("enhanced_pipeline")

    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")
    print(f"✅ Using device: {device}")

    try:
        # Step 1: Load config
        logger.info("Step 1: Initializing configuration")
        config = ISLConfig()

        # Step 2: Preprocess data using the enhanced pipeline
        logger.info("Step 2: Running enhanced data preprocessing")
        preprocessor = EnhancedDataPreprocessor(config)
        manifest, class_names = preprocessor.load_and_preprocess_data()

        config.NUM_CLASSES = len(class_names)
        logger.info(f"📊 Number of classes: {config.NUM_CLASSES}")

        # Step 3: Load DataLoaders with augmentation
        logger.info("Step 3: Loading DataLoaders with augmentation")
        train_loader, test_loader = preprocessor.load_data_for_training(
            batch_size=config.BATCH_SIZE,
            num_workers=4,
            shuffle=True
        )

        # Sanity check
        logger.info("Step 4: Checking data batch shapes")
        for batch_idx, (data, target) in enumerate(train_loader):
            frames, landmarks = data
            print(f"📦 Batch {batch_idx}:")
            print(f"   🖼️ Frames shape: {frames.shape}")       # (B, T, C, H, W)
            print(f"   🧠 Landmarks shape: {landmarks.shape}") # (B, T, 170)
            print(f"   🎯 Labels shape: {target.shape}")
            break

        # Step 5: Build the enhanced model
        logger.info("Step 5: Building Enhanced ISL model")
        model_builder = ISLModelBuilder(config)
        model = model_builder.create_model(num_classes=config.NUM_CLASSES)
        model = model.to(device)

        # Test model forward pass
        logger.info("Testing model forward pass with a batch...")
        with torch.no_grad():
            frames, landmarks = frames.to(device), landmarks.to(device)
            output = model(frames, landmarks)
            print(f"🧮 Model output shape: {output.shape}")

        # Step 6: Setup trainer
        logger.info("Step 6: Initializing Enhanced Trainer")
        trainer = EnhancedISLTrainer(model=model, config=config, device=device)

        optimizer = model_builder.get_optimizer(model)
        total_steps = config.EPOCHS * len(train_loader)
        scheduler = model_builder.get_scheduler(optimizer, total_steps)

        # Step 7: Train the model
        logger.info("Step 7: Starting training loop")
        trainer.train(
            train_loader=train_loader,
            val_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler
        )

        # Step 8: Evaluate the model
        logger.info("Step 8: Running evaluation")
        evaluator = ISLEvaluator(
            model=model,
            test_loader=test_loader,
            device=device,
            class_names=class_names,
            save_dir="detailed_evaluation_results"
        )
        evaluation_data = evaluator.generate_report()

        # Step 9: Log results
        logger.info("===== EVALUATION SUMMARY =====")
        overall = evaluation_data["overall_metrics"]
        logger.info(f"Accuracy: {overall['test_accuracy']:.4f}")
        logger.info(f"Total Test Samples: {overall['total_test_samples']}")
        logger.info(f"Number of Classes: {overall['number_of_classes']}")

        per_class = evaluation_data["per_class_detailed_analysis"]

        # Log top-5 worst classes
        logger.info("=== 5 WORST PERFORMING CLASSES ===")
        worst = sorted(per_class.items(), key=lambda x: x[1]['accuracy'])[:5]
        for cls, info in worst:
            logger.info(f"{cls}: {info['correct_predictions']}/{info['total_test_videos']} "
                        f"({info['accuracy_percentage']:.1f}%)")

        # Log top-5 best classes
        logger.info("=== 5 BEST PERFORMING CLASSES ===")
        best = sorted(per_class.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:5]
        for cls, info in best:
            logger.info(f"{cls}: {info['correct_predictions']}/{info['total_test_videos']} "
                        f"({info['accuracy_percentage']:.1f}%)")

        logger.info("🎉 Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"🔥 Pipeline failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
