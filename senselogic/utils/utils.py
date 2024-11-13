# senselogic/utils/utils.py
import time
import json
import logging
import pandas as pd
import torch
import torch.nn.utils.rnn
import uuid  # Add this import statement
from senselogic.utils.config import CONFIG

logger = logging.getLogger(__name__)


def preprocess_data(row, tokenizer):
    """
    Prepares a single row of data for model input by tokenizing the observations and hypothesis fields.
    Constructs the input sequence by combining observations, and uses the labeled hypothesis as the target.
    """
    logger.debug("Preprocessing data row...")

    try:
        # Define separator token for the T5 model
        separator_token = "</s>"

        # Combine observations into the input sequence
        input_sequence = f"{row['obs1']} {separator_token} {row['obs2']}"
        print(f"Input Sequence (Observations): {input_sequence}")  # Debugging print

        # Select the target hypothesis based on the label
        if row['label'] == "1":
            target_hypothesis = row['hyp1']
        elif row['label'] == "2":
            target_hypothesis = row['hyp2']
        else:
            raise ValueError(f"Invalid label in row: {row['label']}")

        print(f"Target Hypothesis (Label {row['label']}): {target_hypothesis}")  # Debugging print

        # Tokenize input and target sequences
        tokenized_inputs = tokenizer.encode_plus(
            input_sequence, truncation=True, return_tensors="pt", max_length=CONFIG["max_length"]
        )
        print(f"Tokenized Input IDs: {tokenized_inputs['input_ids']}")  # Debugging print
        print(f"Attention Mask: {tokenized_inputs['attention_mask']}")  # Debugging print

        tokenized_targets = tokenizer.encode_plus(
            target_hypothesis, truncation=True, return_tensors="pt", max_length=CONFIG["max_length"]
        )
        print(f"Tokenized Target IDs: {tokenized_targets['input_ids']}")  # Debugging print

        # Return tokenized inputs and labels
        return {
            'input_ids': tokenized_inputs['input_ids'].squeeze(0),
            'attention_mask': tokenized_inputs['attention_mask'].squeeze(0),
            'labels': tokenized_targets['input_ids'].squeeze(0)
        }
    except KeyError as e:
        logger.error(f"Missing key in data row: {e}")
        print(f"Exception encountered due to missing key: {e}")  # Debugging print
        return None
    except Exception as e:
        logger.error(f"Error in preprocess_data: {e}")
        print(f"Exception encountered: {e}")  # Debugging print
        return None


def collate_fn(batch, pad_token_id=0, attention_pad_value=0):
    """
    Collates a batch of preprocessed data into a format suitable for model input,
    including padding to equalize the lengths of sequences within the batch.
    """
    print("\n--- Collating Batch ---")  # Debugging print

    # Unpack the batch into separate lists for each field.
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Padding sequences for 'input_ids', 'attention_masks', and 'labels'
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=attention_pad_value)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_token_id)

    # Debug prints to inspect the shapes and contents of padded tensors
    print(f"Input IDs Padded Shape: {input_ids_padded.shape}")  # Shape debugging print
    print(f"Input IDs Padded: {input_ids_padded}")  # Content debugging print
    print(f"Attention Mask Padded Shape: {attention_masks_padded.shape}")  # Shape debugging print
    print(f"Attention Mask Padded: {attention_masks_padded}")  # Content debugging print
    print(f"Labels Padded Shape: {labels_padded.shape}")  # Shape debugging print
    print(f"Labels Padded: {labels_padded}")  # Content debugging print

    # Return the padded tensors
    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'labels': labels_padded
    }

