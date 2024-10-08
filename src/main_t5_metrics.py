import os
import pandas as pd
import logging
from src.utils.metrics import MetricsEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory where metrics files will be saved
metrics_output_dir = '/data/agirard/Projects/TimeTravel-PolicyGradientRL/results/metrics'
os.makedirs(metrics_output_dir, exist_ok=True)  # Create the directory if it doesn't exist

def process_epoch_data(df, epoch):
    """
    Function to calculate and return similarity metrics for Epoch 5 or 6.
    """
    # Filter data for the specific epoch (5 or 6)
    epoch_data = df[df['Epoch'] == epoch]

    # Extract necessary columns
    generated_texts = epoch_data['Generated Text'].tolist()
    edited_endings = epoch_data['Edited Ending'].tolist()
    counterfactuals = epoch_data['Counterfactual'].tolist()
    initials = epoch_data['Initial'].tolist()
    premises = epoch_data['Premise'].tolist()
    original_endings = epoch_data['Original Ending'].tolist()
    
    # Initialize the MetricsEvaluator
    evaluator = MetricsEvaluator()

    # Calculate the metrics only for Epoch 5 or 6
    all_metrics = {}
    
    # Calculate BART similarity
    all_metrics.update(evaluator.calculate_and_log_bart_similarity(
        generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
    ))
    
    # Calculate BERT similarity
    all_metrics.update(evaluator.calculate_and_log_bert_similarity(
        generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
    ))
    
    # Calculate BLEU scores
    all_metrics.update(evaluator.calculate_and_log_bleu_scores(
        generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
    ))
    
    # Calculate ROUGE scores
    all_metrics.update(evaluator.calculate_and_log_rouge_scores(
        generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
    ))

    # Convert metrics to DataFrame and transpose
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index', columns=[f'Epoch {epoch}'])
    metrics_df.reset_index(inplace=True)
    metrics_df.columns = ['Metric', f'Epoch {epoch}']

    return metrics_df

def process_file(file_path, model_id):
    """
    Process a specific file (either validation or test) for the given model ID.
    """
    if os.path.exists(file_path):
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Initialize an empty DataFrame to store combined metrics
        combined_metrics_df = pd.DataFrame()

        # Get unique epochs
        epochs = df['Epoch'].unique()

        # Filter epochs for only Epoch 5 and Epoch 6
        epochs_to_process = [epoch for epoch in epochs if epoch in [5, 6]]

        # Process each filtered epoch
        for epoch in epochs_to_process:
            epoch_metrics_df = process_epoch_data(df, epoch)
            if combined_metrics_df.empty:
                combined_metrics_df = epoch_metrics_df
            else:
                combined_metrics_df = pd.merge(combined_metrics_df, epoch_metrics_df, on='Metric', how='outer')
        
        # Determine the base name of the file (validation_details or test_details)
        base_name = os.path.basename(file_path).replace('.csv', '')

        # Generate a new file path in the metrics output directory, including the model ID
        file_name = f'{base_name}_{model_id}_metrics.csv'
        metrics_file_path = os.path.join(metrics_output_dir, file_name)
        
        # Save the combined metrics to a CSV file
        combined_metrics_df.to_csv(metrics_file_path, index=False)
        print(f"Combined metrics saved to {metrics_file_path}")
    else:
        print(f"File not found: {file_path}")

def main():
    """
    Main function to process the CSV files for all specified models.
    """
    model_ids = [
        # T5-base models (Test data)
        '2024-03-22-10',  # Weight: 1-1
        '2024-09-02-21',  # Weight: 5-1
        '2024-09-03-06',  # Weight: 10-1
        '2024-04-09-11',  # Weight: 1-12
        '2024-04-09-22',  # Weight: 1-13
        '2024-09-03-08',  # Weight: 15-1
        '2024-04-08-13',  # Weight: 20-1
        '2024-09-03-11',  # Weight: 25-1
        '2024-09-03-15',  # Weight: 30-1

        # T5-large models (Test data)
        '2024-03-22-15',  # Weight: 1-1
        '2024-08-29-14',  # Weight: 5-1
        '2024-08-28-20',  # Weight: 10-1
        '2024-04-10-10',  # Weight: 15-1
        '2024-04-08-09',  # Weight: 20-1
        '2024-09-02-14',  # Weight: 25-1
        '2024-04-10-14',  # Weight: 30-1

        # T5-base models (Gold data)
        '2024-09-03-17',  # Weight: 5-1
        '2024-09-03-20',  # Weight: 10-1
        '2024-09-04-12',  # Weight: 12-1
        '2024-08-15-10',  # Weight: 13-1
        '2024-09-04-01',  # Weight: 15-1
        '2024-09-04-06',  # Weight: 20-1
        '2024-09-04-08',  # Weight: 25-1
        '2024-09-04-10',  # Weight: 30-1

        # T5-large models (Gold data)
        '2024-08-30-11',  # Weight: 5-1
        '2024-08-30-06',  # Weight: 10-1
        '2024-08-29-21',  # Weight: 15-1
        '2024-08-15-20',  # Weight: 20-1
        '2024-09-02-09',  # Weight: 25-1
        '2024-08-30-16'   # Weight: 30-1
    ]
    
    for model_id in model_ids:
        base_path = f'/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/model_{model_id}'
        
        # Process validation details file
        validation_file_path = os.path.join(base_path, 'validation_details.csv')
        process_file(validation_file_path, model_id)
        
        # Process test details file
        test_file_path = os.path.join(base_path, 'test_details.csv')
        process_file(test_file_path, model_id)

if __name__ == "__main__":
    main()
