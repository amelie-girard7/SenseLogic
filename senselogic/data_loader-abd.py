# data_loader-abd.py
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from senselogic.utils.utils-abd import preprocess_data, collate_fn

class CustomJSONDataset(Dataset):
    """
    A custom PyTorch Dataset class designed for loading and preprocessing data stored in JSONL format.
    Supports tokenization and preprocessing for model training and evaluation.
    """
    def __init__(self, file_path, tokenizer):
        """
        Initializes the dataset object.

        Args:
            file_path (str): Path to the JSONL file containing the data.
            tokenizer (T5Tokenizer): The tokenizer to use for preprocessing.
        """
        print(f"Attempting to load data from: {file_path}")  # Debug print for loading path
        
        # Attempt to load and preprocess the data from the provided JSONL file
        try:
            # Specify lines=True to correctly read JSONL format
            data = pd.read_json(file_path, lines=True)
            print(f"Loaded data from {file_path} with {len(data)} lines.")  # Debug print
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing {file_path}: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        self.tokenizer = tokenizer

        # Preprocess each row of the data using the provided tokenizer
        self.processed_data = data.apply(lambda row: preprocess_data(row, self.tokenizer), axis=1, result_type='expand')
        print("Sample of processed data:")  # Debug print to check data structure
        print(self.processed_data.head())

    def __len__(self):
        """Returns the total number of items in the dataset."""
        return len(self.processed_data)

    def __getitem__(self, idx):
        """
        Retrieves an item by its index from the dataset.
        
        Args:
            idx (int): The index of the item to retrieve.
        
        Returns:
            dict: A single data item, preprocessed and ready for model input.
        """
        item = self.processed_data.iloc[idx]
        print(f"Retrieved item {idx} with keys: {list(item.keys())}")  # Debug print
        return item

def create_dataloaders(file_path, tokenizer, batch_size, num_workers):
    """
    Creates a DataLoader instance for the specified dataset file.

    Args:
        file_path (str): Full path to the data file.
        tokenizer (T5Tokenizer): The tokenizer to use for preprocessing the data.
        batch_size (int): The number of samples per batch.
        num_workers (int): The number of worker threads to use for loading data.

    Returns:
        DataLoader: A DataLoader object for the specified dataset.
    """
    print(f"Checking file: {file_path}")  # Debug print
    if not Path(file_path).exists():
        raise FileNotFoundError(f"{file_path} does not exist.")
    
    # Create an instance of the dataset for the specified data file
    dataset = CustomJSONDataset(file_path, tokenizer)

    # Determine whether to shuffle: only shuffle for training
    shuffle = "train" in file_path
    print(f"Creating DataLoader with shuffle={shuffle} for training" if shuffle else "for validation/testing")  # Debug print

    # Create and return the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id=tokenizer.pad_token_id),
        num_workers=num_workers,
        shuffle=shuffle  # Shuffle data only for training
    )

    print(f"Created DataLoader for {file_path} with batch_size={batch_size}, shuffle={shuffle}")  # Debug print
    return dataloader
