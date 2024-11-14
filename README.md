### Project Overview

This project leverages Natural Language Generation (NLG) to generate coherent narrative endings based on a context of observations. Utilizing datasets like ART and TimeTravel, which are typically used for hypothesis selection, we adapt them for NLG. Here, instead of selecting from predefined hypotheses, the model learns to generate a contextually appropriate ending based on the observations. We achieve this by treating one of the hypotheses (hyp1 or hyp2) as a target label in a supervised training setup.

### Goals and Approach

1. **Dataset Adaptation for NLG**:
   - In the ART and TimeTravel datasets, each sample includes:
     - Two observations (`obs1` and `obs2`) providing context.
     - Two hypotheses (`hyp1` and `hyp2`) as potential narrative continuations.
     - A label indicating the correct continuation (either `1` or `2`, referring to `hyp1` or `hyp2`).
   - We adapt this structure by framing the task as a generation problem rather than a selection problem:
     - The model receives `obs1` and `obs2` as input.
     - The correct hypothesis (either `hyp1` or `hyp2` based on the label) becomes the target text for the model to generate.

2. **Supervised Learning with Maximum Likelihood Estimation (MLE)**:
   - In the **MLE phase**, we train the model to learn the correct narrative continuation by using cross-entropy loss. Specifically:
     - During each training step, `obs1` and `obs2` are fed to the model, and the labeled hypothesis (`hyp1` or `hyp2`) serves as the target output.
     - The model is thus trained in a supervised fashion to produce a generated output that matches the labeled hypothesis as closely as possible.
   - This MLE phase allows the model to learn foundational narrative generation patterns, ensuring that it generates endings that align with the given context.

3. **Policy Gradient (PG) Fine-Tuning for Enhanced Generation Quality**:
   - Following the MLE training phase, we use **Policy Gradient (PG)** fine-tuning to further optimize the model’s ability to generate high-quality, contextually appropriate endings.
   - The PG phase introduces a reward-based learning component:
     - The model generates an ending based on `obs1` and `obs2`.
     - A reward function evaluates the generated ending on various metrics, including **ROUGE**, **BERTScore**, **BARTScore**, and **BLEU**. These metrics assess fluency, relevance, and narrative coherence.
     - Specifically:
       - **ROUGE** evaluates the overlap between generated and reference endings, focusing on recall and precision.
       - **BERTScore** compares embeddings of the generated and target text, capturing semantic similarity.
       - **BARTScore** (based on BART language model scoring) assesses the fluency and overall quality of the generated text.
       - **BLEU** provides a measure of precision based on word overlap, commonly used in NLG for matching n-grams.
     - Using these metrics, the reward function assigns a composite score to the generated ending, promoting those that balance coherence, relevance, and fluency.
     - The model updates its generation strategy to maximize these rewards, thereby refining its ability to produce higher-quality, contextually appropriate endings over time.
   - This PG-based fine-tuning phase enables the model to learn more nuanced, sophisticated generation patterns, surpassing the limitations of supervised learning alone and enhancing its storytelling capabilities.

4. **Reward Functions and Evaluation Metrics**:
   - In the PG phase, reward functions assess generated endings for qualities like fluency, coherence, and adherence to the narrative structure.
   - Generated texts are evaluated using NLG metrics (e.g., BLEU, ROUGE, BERTScore, BARTScore) as well as task-specific evaluations (e.g., human assessments of narrative coherence and accuracy).

### Example Workflow

Given input observations:
- `obs1`: "Stephen was at a party."
- `obs2`: "He checked it but it was completely broken."

If the label is `2`, the model is trained to generate `hyp2`:
- `hyp2`: "Stephen knocked over a vase while drunk."

During training, the model will learn to generate text that mirrors `hyp2` based on the context provided by `obs1` and `obs2`.

### Summary

This project transforms binary-choice tasks into open-ended generation tasks. Using MLE, the model learns to generate narrative endings that correspond to labeled hypotheses. PG fine-tuning further enhances the model's outputs by introducing a reward function based on NLG metrics like ROUGE, BERTScore, BARTScore, and BLEU. This reward mechanism optimizes the model for narrative quality, coherence, and relevance. By combining these training methods, the project shifts the dataset's original selection-based format into a creative NLG task, achieving richer and more coherent storytelling.