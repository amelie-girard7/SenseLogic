# senselogic/utils/metrics-abd.py
import logging
from sacrebleu.metrics import BLEU
from rouge import Rouge
from bert_score import BERTScorer
import torch
from senselogic.utils.config-abd import CONFIG
from senselogic.BARTScore_metric.bart_score import BARTScorer

logger = logging.getLogger(__name__)

class MetricsEvaluator:
    """
    A class for evaluating text generation models using various metrics such as BLEU, ROUGE, BERTScore, and BARTScore.
    This class helps compute scores for supervised learning and rewards for reinforcement learning.
    """

    def __init__(self):
        """
        Initializes the metric evaluators based on the configurations provided in CONFIG.
        Depending on the settings, it initializes evaluators for BLEU, ROUGE, BERTScore, and BARTScore.
        """
        print(f"Initializing MetricsEvaluator with config: {CONFIG}")
        
        # Initialize ROUGE for ROUGE score calculation
        self.rouge = Rouge()  

        # Initialize BERTScorer if configured to use BERTScore
        self.bert_scorer = BERTScorer(
            model_type=CONFIG.get("bert_scorer_model_type", "bert-base-uncased"),
            device=CONFIG.get("scorer_device", "cpu"),
            num_layers=CONFIG.get("bert_scorer_num_layers", None),
            batch_size=CONFIG.get("bert_scorer_batch_size", 16)
        ) if CONFIG.get("use_bert", False) else None

        # Initialize BLEU scorer if configured to use BLEU
        self.sacre_bleu = BLEU() if CONFIG.get("use_bleu", False) else None

        # Check if sacrebleu supports effective_order
        self.supports_effective_order = hasattr(self.sacre_bleu, 'sentence_score') and \
                                        'effective_order' in BLEU.sentence_score.__code__.co_varnames

        # Initialize BARTScorer if configured to use BARTScore
        self.bart_scorer = BARTScorer(
            device=CONFIG.get("scorer_device", "cpu"), 
            checkpoint=CONFIG.get("bart_scorer_checkpoint", "facebook/bart-large-cnn")
        ) if CONFIG.get("use_bart", False) else None

        print("MetricsEvaluator initialized.")

    def calculate_score(self, generated_texts, references):
        """
        Calculates the score based on the specified metric in CONFIG.
        This method supports ROUGE-L, BERTScore, BLEU, and BARTScore as scoring mechanisms.
        
        Args:
            generated_texts (list of str): Texts generated by the model.
            references (list of str): Reference texts (ground truth).

        Returns:
            scores_tensor (torch.Tensor): A tensor of score values per example on the specified device.
        """
        score_metric = CONFIG.get("reward_metric", "rouge")
        scorer_device = CONFIG.get("scorer_device", "cpu")

        # Debugging prints for input and metric selection
        print(f"Score metric: {score_metric}")
        print(f"Generated texts: {generated_texts}")
        print(f"Reference texts: {references}")

        # Ensure inputs are lists of strings
        generated_texts = [str(gt) for gt in generated_texts]
        references = [str(ref) for ref in references]

        # Case 1: ROUGE-L is selected as the score metric
        if score_metric == "rouge":
            print("Calculating ROUGE-L scores...")
            rouge_scores = self.rouge.get_scores(generated_texts, references, avg=False)
            scores = [score['rouge-l']['f'] for score in rouge_scores]  # Extract ROUGE-L F1 scores

        # Case 2: BERTScore is selected as the score metric
        elif score_metric == "bert":
            if self.bert_scorer is None:
                raise ValueError("BERTScore is not initialized. Set 'use_bert' to True in CONFIG.")
            print("Calculating BERTScore...")
            _, _, f1 = self.bert_scorer.score(generated_texts, references)
            scores = f1.tolist()  # Convert tensor to list

        # Case 3: BLEU is selected as the score metric with one-to-one comparison
        elif score_metric == "bleu":
            if self.sacre_bleu is None:
                raise ValueError("BLEU scorer is not initialized. Set 'use_bleu' to True in CONFIG.")
            print("Calculating corpus-level BLEU score...")

            # SacreBLEU corpus score calculation, where the references need to be nested as a list of lists
            references = [[ref] for ref in references]  # Corpus-level BLEU expects list of list of references
            bleu_result = self.sacre_bleu.corpus_score(generated_texts, references)
            score = bleu_result.score  # Retrieve the BLEU score for the corpus

            # Print BLEU score for debugging
            print(f"Corpus-level BLEU score: {score}")
            scores = [score] * len(generated_texts)  # Replicate for each item in batch for compatibility


        # Case 4: BARTScore is selected as the score metric
        elif score_metric == "bart":
            if self.bart_scorer is None:
                raise ValueError("BARTScore is not initialized. Set 'use_bart' to True in CONFIG.")
            print("Calculating BARTScore...")
            scores = self.bart_scorer.score(generated_texts, references)  # BARTScore for each example

        # Unsupported score metric
        else:
            raise ValueError(f"Unsupported score metric: {score_metric}")

        # Convert scores to a tensor and move it to the specified device
        scores_tensor = torch.tensor(scores, dtype=torch.float32, device=scorer_device)
        print(f"Scores tensor (on {scorer_device}): {scores_tensor}")  # Print the final scores tensor
        return scores_tensor

    def calculate_and_log_rouge_scores(self, all_generated_texts, all_edited_endings, all_counterfactuals, all_initials, all_premises, all_original_endings, logger):
        """
        Calculates and logs ROUGE scores for various comparisons between generated texts and references.
        """
        print("Calculating ROUGE scores...")

        all_comparisons = [
            ('rouge_prediction_edited', all_generated_texts, all_edited_endings),
            ('rouge_prediction_cf', all_generated_texts, all_counterfactuals),
            ('rouge_prediction_initial', all_generated_texts, all_initials),
            ('rouge_prediction_original', all_generated_texts, all_original_endings),
            ('rouge_edited_ending_cf', all_edited_endings, all_counterfactuals),
            ('rouge_edited_ending_initial', all_edited_endings, all_initials),
            ('rouge_edited_ending_original', all_edited_endings, all_original_endings),
        ]

        rouge_scores = {}
        for label, hypotheses, references in all_comparisons:
            if references:
                try:
                    rouge_scores_set = self.rouge.get_scores(hypotheses, references, avg=True)
                    score_type = 'rouge-l'
                    rouge_scores[f"{label}_{score_type}_f"] = rouge_scores_set[score_type]['f']
                    logger.info(f"{label}_{score_type}_f: {rouge_scores_set[score_type]['f']}")
                except Exception as e:
                    logger.error(f"Error calculating {label}: {e}")
                    rouge_scores[f"{label}_f"] = 'N/A'

        return rouge_scores

    def calculate_and_log_bert_similarity(self, all_generated_texts, all_edited_endings, all_counterfactuals, all_initials, all_premises, all_original_endings, logger):
        """
        Calculates and logs BERT similarity F1 scores for various comparisons of generated texts and references.
        """
        print("Calculating BERT similarity F1 scores...")

        all_comparisons = [
            ('bert_prediction_edited', all_generated_texts, all_edited_endings),
            ('bert_prediction_cf', all_generated_texts, all_counterfactuals),
            ('bert_prediction_initial', all_generated_texts, all_initials),
            ('bert_prediction_original', all_generated_texts, all_original_endings),
            ('bert_edited_ending_cf', all_edited_endings, all_counterfactuals),
            ('bert_edited_ending_initial', all_edited_endings, all_initials),
            ('bert_edited_ending_original', all_edited_endings, all_original_endings),
        ]

        bert_scores = {}
        for label, texts_a, texts_b in all_comparisons:
            if texts_b:
                try:
                    _, _, f1 = self.bert_scorer.score(texts_a, texts_b)
                    avg_f1 = f1.mean().item()
                    logger.info(f"{label}_f1: {avg_f1}")
                    bert_scores[f"{label}_f1"] = avg_f1
                except Exception as e:
                    logger.error(f"Error calculating {label}: {e}")
                    bert_scores[f"{label}_f1"] = 'N/A'

        return bert_scores

    def calculate_and_log_bart_similarity(self, all_generated_texts, all_edited_endings, all_counterfactuals, all_initials, all_premises, all_original_endings, logger):
        """
        Calculates and logs BART-based similarity scores for a variety of text comparisons,
        using the BARTScorer to evaluate the similarity between different segments of texts.
        """
        print("Calculating BART similarity scores...")
        
        all_comparisons = [
            ('bart_prediction_edited', all_generated_texts, all_edited_endings),
            ('bart_prediction_cf', all_generated_texts, all_counterfactuals),
            ('bart_prediction_initial', all_generated_texts, all_initials),
            ('bart_prediction_original', all_generated_texts, all_original_endings),
            ('bart_edited_ending_cf', all_edited_endings, all_counterfactuals),
            ('bart_edited_ending_initial', all_edited_endings, all_initials),
            ('bart_edited_ending_original', all_edited_endings, all_original_endings),
        ]

        bart_scores = {}
        for label, src_texts, tgt_texts in all_comparisons:
            if tgt_texts:
                try:
                    scores = self.bart_scorer.score(src_texts, tgt_texts, batch_size=4)
                    avg_score = sum(scores) / len(scores) if scores else float('nan')
                    logger.info(f"{label}_avg_score: {avg_score}")
                    bart_scores[f"{label}_avg_score"] = avg_score
                    print(f"{label}_avg_score: {avg_score}")
                except Exception as e:
                    logger.error(f"Error calculating {label}: {e}")
                    bart_scores[f"{label}_avg_score"] = 'N/A'
                    print(f"Error calculating {label}: {e}")

        return bart_scores

    def calculate_and_log_bleu_scores(self, all_generated_texts, all_edited_endings, all_counterfactuals, all_initials, all_premises, all_original_endings, logger):
        """
        Calculates and logs SacreBLEU scores for various comparisons between generated texts and references.
        
        Args:
        - all_generated_texts: List of generated texts
        - all_edited_endings: List of reference edited endings
        - all_counterfactuals: List of counterfactuals
        - all_initials: List of initial events
        - all_premises: List of premises
        - all_original_endings: List of original endings
        - logger: Logger for logging BLEU score information

        Returns:
        - bleu_scores: Dictionary with calculated BLEU scores for different comparisons
        """
        print("Calculating BLEU scores...")

        # Prepare references for BLEU score calculation
        edited_endings_refs = [[ending] for ending in all_edited_endings] if all_edited_endings else None
        counterfactuals_refs = [[cf] for cf in all_counterfactuals]
        initials_refs = [[init] for init in all_initials]
        original_endings_refs = [[orig] for orig in all_original_endings]

        # List of all comparisons we want to calculate BLEU scores for
        all_comparisons = [
            ('bleu_prediction_edited', all_generated_texts, edited_endings_refs),
            ('bleu_prediction_cf', all_generated_texts, counterfactuals_refs),
            ('bleu_prediction_initial', all_generated_texts, initials_refs),
            ('bleu_prediction_original', all_generated_texts, original_endings_refs),
            ('bleu_edited_ending_cf', all_edited_endings, counterfactuals_refs),
            ('bleu_edited_ending_initial', all_edited_endings, initials_refs),
            ('bleu_edited_ending_original', all_edited_endings, original_endings_refs),
        ]

        # Dictionary to store BLEU scores for each comparison
        bleu_scores = {}
        for label, texts, references in all_comparisons:
            if references is not None:
                try:
                    # Calculate BLEU score
                    bleu_result = self.sacre_bleu.corpus_score(texts, references)
                    bleu_score = bleu_result.score
                    logger.info(f"{label}: {bleu_score}")
                    bleu_scores[label] = bleu_score
                except Exception as e:
                    logger.error(f"Error calculating {label}: {e}")
                    bleu_scores[label] = 'N/A'

        return bleu_scores