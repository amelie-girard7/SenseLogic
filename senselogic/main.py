# main.py
import os
import datetime
import logging
from transformers import T5Tokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from senselogic.models.model import FlanT5FineTuner
from senselogic.data_loader import create_dataloaders
from senselogic.utils.config import CONFIG

# Add the project root to PYTHONPATH



# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_model(model_dir, file_label="", checkpoint_path=None, use_policy_gradient=False):
    """
    Initializes the model, optionally loading from a checkpoint.
    """
    if checkpoint_path:
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        model = FlanT5FineTuner.load_from_checkpoint(
            checkpoint_path,
            model_name=CONFIG["model_name"],
            model_dir=model_dir,
            file_label=file_label
        )
    else:
        logger.info(f"Initializing a fresh model: {CONFIG['model_name']} with label {file_label}")
        model = FlanT5FineTuner(CONFIG["model_name"], model_dir, file_label=file_label)

    # Set model training mode
    model.use_policy_gradient = use_policy_gradient
    return model

def setup_trainer(model_dir, max_epochs, checkpoint_callback, wandb_logger, starting_epoch=0):
    """
    Sets up the PyTorch Lightning Trainer with W&B logger and checkpointing.
    """
    trainer = Trainer(
        max_epochs=max_epochs + starting_epoch,
        accelerator='gpu',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        val_check_interval=0.1
    )
    logger.info(f"Trainer setup complete for {max_epochs} epochs starting from epoch {starting_epoch}.")
    return trainer

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Unique directory based on timestamp
    model_timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H")
    model_dir = CONFIG["models_dir"] / f"experiment_{model_timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Setup WandB logger with debug information
    print(f"Initializing WandB logger with project: 'abductiveStory', entity: 'counterfactualStory'")
    wandb_logger = WandbLogger(
        project="abductiveStory",
        entity="counterfactualStory",
        log_model="all"
    )
    wandb_logger.experiment.config.update(CONFIG)
    
    # Setup tokenizer
    tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_name"], legacy=False)
    
    # Correct paths for train, dev, and test datasets
    train_file_path = "/data/agirard/Projects/SenseLogic/data/anlg/train-w-comet-preds-sample.jsonl"
    dev_file_path = "/data/agirard/Projects/SenseLogic/data/anlg/dev-w-comet-preds-sample.jsonl"
    test_file_path = "/data/agirard/Projects/SenseLogic/data/anlg/test-w-comet-preds-sample.jsonl"
    
    # Print hardcoded paths for confirmation
    print(f"Using hardcoded paths:")
    print(f"Train file path: {train_file_path}")
    print(f"Dev file path: {dev_file_path}")
    print(f"Test file path: {test_file_path}")

    # Create dataloaders with correct paths
    dataloaders = {
        "train": create_dataloaders(train_file_path, tokenizer, CONFIG["batch_size"], CONFIG["num_workers"]),
        "dev": create_dataloaders(dev_file_path, tokenizer, CONFIG["batch_size"], CONFIG["num_workers"]),
        "test": create_dataloaders(test_file_path, tokenizer, CONFIG["batch_size"], CONFIG["num_workers"]),
    }

    model = None  # Initialize model variable to track if any phase trained a model

    # --- MLE Phase ---
    if CONFIG["mle_enabled"]:
        mle_checkpoint = CONFIG["mle_checkpoint_path"] if CONFIG["mle_from_checkpoint"] else None
        print("Starting MLE phase training...")
        model = setup_model(
            model_dir, 
            file_label="_mle", 
            checkpoint_path=mle_checkpoint, 
            use_policy_gradient=False
        )

        mle_checkpoint_callback = ModelCheckpoint(
            dirpath=model_dir,
            monitor='validation_mle_loss',
            mode='min',
            save_top_k=1,
            filename="best_mle_checkpoint"
        )

        # Train with MLE
        trainer = setup_trainer(model_dir, CONFIG["mle_epochs"], mle_checkpoint_callback, wandb_logger)
        trainer.fit(model, dataloaders["train"], dataloaders["dev"])
        
        # Capture the last completed epoch in MLE phase for continuity in PG phase
        last_mle_epoch = trainer.current_epoch
        print(f"MLE training completed. Last MLE epoch: {last_mle_epoch}")

    # --- PG Phase ---
    if CONFIG["pg_enabled"]:
        print("Starting PG phase training...")
        
        # Load from PG-specific checkpoint or latest MLE checkpoint
        pg_checkpoint = CONFIG["pg_checkpoint_path"] if CONFIG["pg_from_checkpoint"] else mle_checkpoint_callback.best_model_path
        model = setup_model(
            model_dir, 
            file_label="_pg", 
            checkpoint_path=pg_checkpoint, 
            use_policy_gradient=True
        )

        pg_checkpoint_callback = ModelCheckpoint(
            dirpath=model_dir,
            monitor='validation_pg_loss',
            mode='max',
            save_top_k=1,
            filename="best_pg_checkpoint"
        )

        # Set the starting epoch for PG phase to continue logging from MLE phase
        trainer = setup_trainer(model_dir, CONFIG["pg_epochs"], pg_checkpoint_callback, wandb_logger, starting_epoch=last_mle_epoch)
        
        # Start PG training from the MLE checkpoint
        trainer.fit(model, dataloaders["train"], dataloaders["dev"], ckpt_path=pg_checkpoint)
        print(f"PG training completed from epoch {last_mle_epoch} onward.")

    # --- Testing Phase ---
    if model:
        logger.info("Testing the final model.")
        trainer.test(model, dataloaders["test"])
    else:
        logger.info("No model was trained, skipping testing.")

if __name__ == '__main__':
    logger.info("Starting the main process...")
    main()
    logger.info("Process completed.")
