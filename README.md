# LLM Extraction of Interpretable Features from Text

This repository contains the code and supplementary materials for the scientific article **"LLM Extraction of Interpretable Features from Text."** The aim of this project is to demonstrate how large language models (LLMs) can be used to extract interpretable features from textual data. We further demonstrate the use of these interpretable features with action rules.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Experiments and Notebooks](#experiments)
- [Results](#results)
- [License](#license)

## Introduction
Existing text representations such as embeddings and bag-of-words are not suitable for rule learning due to their high dimensionality and absent or questionable feature-level interpretability. This article explores whether large language models (LLMs) could address this by extracting a small number of interpretable features from text.
We demonstrate this process on two datasets (CORD-19  and M17+) containing several thousand scientific articles from multiple disciplines and a target being a proxy for research impact. An evaluation based on testing for the statistically significant correlation with research impact has shown that LLama 2-generated features are semantically meaningful. We consequently used these generated features in text classification to predict the binary target variable representing the citation rate for the CORD-19 dataset and the ordinal 5-class target representing an expert-awarded grade in the M17+ dataset. Machine-learning models trained on the LLM-generated features provided similar predictive performance to the state-of-the-art embedding model SciBERT for scientific text. Not only did the LLM use only 62 features compared to 768 features in SciBERT embeddings, but these features were fully directly interpretable, as they corresponded to notions such as article methodological rigour, novelty, or grammatical correctness. Consequently, we apply action rule mining resulting in a small number of well-interpretable rules. Consistently competitive results obtained with the same LLM feature set across both thematically diverse datasets show that this approach generalizes across domains.  We also assume this technique could be used not only in rule learning but also in other white-box methods. Our results are replicable due to the use of open LLM. 

## Installation

To get started, clone this repository and install the necessary dependencies:

```bash
git clone https://github.com/vojtech-balek/llm-features.git
cd llm-features
pip install -r requirements.txt
```

## Experiments

### Data
Data is stored in [data](data) folder.

### Feature Extraction

Feature extraction with user-defined features as described in Methodology. The corresponding notebook is [ffeature_extraction-CORD19-M17plus.ipynb](notebooks/feauture_extraction/feature_extraction-CORD19-M17plus.ipynb).

Feature extraction with LLM-generated features using Batch API. The corresponding notebook is [llm-chatgpt.ipynb](notebooks/feauture_extraction/llm-chatgpt.ipynb).

### Feature Analysis

Analysis of the features generated for the datasets. Formal test of the relationship between target and generatef features:

* CORD-19 and M17+: [feature_analysis-CORD19-M17plus.ipynb](notebooks/feature_analysis/feature_analysis-CORD19-M17plus.ipynb)
* BANKING77: [feature_analysis-bank77.ipynb](notebooks/feature_analysis/feature_analysis-bank77.ipynb)
* Hate Speech: [feature_analysis-hate_speech.ipynb](notebooks/feature_analysis/feature_analysis-hate_speech.ipynb)

### Model Evaluation

Evaluate the performance of the models and the extracted user-defined features:

* CORD-19 and M17+: [llm_classification-LLAMA2-c19-m17.ipynb](notebooks/classification/llm_classification-LLAMA2-c19-m17.ipynb)

Evaluate the performance of the models and the extracted LLM-generated features:

* BANKING77: [llm-classification-bank77.ipynb](notebooks/classification/llm-classification-bank77.ipynb)
* Hate Speech: [llm-classification-hate.ipynb](notebooks/classification/llm-classification-hate.ipynb)
* Food Hazard: [llm-classification-hazard.ipynb](notebooks/classification/llm-classification-hazard.ipynb)
* CORD-19: [llm-classification-c19.ipynb](notebooks/classification/llm-classification-c19.ipynb)
* M17+: [llm-classification-m17.ipynb](notebooks/classification/llm-classification-m17.ipynb)

### Action Rules

Mining of the action rules for CORD-19 and M17+ datasets [action-CORD19.ipynb](notebooks/action_rules/actions-CORD19.ipynb) and  [action-M17Plus.ipynb](notebooks/action_rules/actions-M17Plus.ipynb).


### License

MIT
