# /data/agirard/Projects/SenseLogic/senselogic/utils/config-abd.py
import os
from pathlib import Path

# Set the root directory based on an environment variable or default to a parent directory
ROOT_DIR = Path(os.getenv('SENSELOGIC_ROOT', Path(__file__).resolve().parent.parent.parent))

# Configuration dictionary for model training, paths, and other settings
CONFIG = {
    # Paths relative to the root directory
    "root_dir": ROOT_DIR,
    "data_dir": ROOT_DIR / "data" / "anlg",  # Directory containing transformed data
    "models_dir": ROOT_DIR / "models",  # Directory to save models
    "logs_dir": ROOT_DIR / "logs",  # Directory for logs
    "results_dir": ROOT_DIR / "results",  # Directory for results (e.g., validation details)

    # Sample File names for training, validation, and test datasets
    "train_file": "train-w-comet-preds.jsonl",
    "dev_file": "dev-w-comet-preds.jsonl",
    "test_file": "test-w-comet-preds.jsonl",

    # File names for training, validation, and test datasets
    #"train_file": "train_supervised_small.json",
    #"dev_file": "dev_data.json",
    #"test_file": "test_data.json",

    # Model and training configurations
    "model_name": os.getenv('MODEL_NAME', "google/flan-t5-base"),  # Hugging Face model to load
    "batch_size": int(os.getenv('BATCH_SIZE', 4)),  # Number of samples per batch
    "num_workers": int(os.getenv('NUM_WORKERS', 3)),  # Number of workers for data loading
    "learning_rate": float(os.getenv('LEARNING_RATE', 2e-5)),  # Learning rate for the optimizer

    # Additional training options
    "use_custom_loss": False,  # Whether to use a custom loss function (set to False for MLE)
    "output_attentions": False,  # Set to True to output attentions from the model (optional)

    # MLE Phase configurations
    "mle_enabled": True,  # Enable MLE training
    "mle_from_checkpoint": False,  # Start training from scratch (no checkpoint)
    "mle_checkpoint_path": None,  # No checkpoint path since we start from scratch
    "mle_epochs": 3,  # Number of epochs to train with MLE

    # PG Phase configurations (disabled in this experiment)
    "pg_enabled": True,  # Disable policy gradient training (PG phase)
    "pg_from_checkpoint": False,  # Start PG training from the best MLE checkpoint, not a separate checkpoint
    "pg_checkpoint_path": None,   # Leave as None to use the best MLE checkpoint
    "pg_epochs": 3,  # Number of epochs to fine-tune with PG

    # Reward-based training configurations
    "reward_metric": "bleu",   ## Primary reward metric for PG (e.g., "rouge", "bert", "bart") (default to "rouge")
    "baseline_score": 0.5,  # Baseline score for PG (used to calculate rewards)

    # Preprocessing and generation parameters
    "max_length": 512,  # Maximum length for input data
    "shuffle": True,  # Shuffle the data during training
    "max_gen_length": 250,  # Maximum length for generated text

    # Additional configuration for scoring metrics 
    "use_bert": False,  # Disable BERT scorer
    "bert_scorer_model_type": "microsoft/deberta-xlarge-mnli",  # Default BERT model for scorer 
    "scorer_device": "cuda:0",  # Device for the scorer
    "bert_scorer_batch_size": 4,  # Batch size for BERT scorer 

    "use_bleu": True,  # Disable BLEU scorer,
       
    "use_bart": False,  # Disable BART scorer
    "bart_scorer_checkpoint": "facebook/bart-large-cnn"  # Default BART model for scorer 
}

# Create any directories that don't exist
for path_key in ['data_dir', 'models_dir', 'logs_dir', 'results_dir']:
    path = CONFIG[path_key]
    if not path.exists():
        print(f"Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
