# LLM Extraction of Interpretable Features from Text

This repository contains the code and supplementary materials for the scientific article **"LLM Extraction of Interpretable Features from Text."** The aim of this project is to demonstrate how large language models (LLMs) can be used to extract interpretable features from textual data. We further demonstrate the use of these interpretable features with action rules.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Experiments and Notebooks](#experiments)
- [Results](#results)
- [License](#license)

## Introduction
Existing text representations such as embeddings and bag-of-words are not suitable for rule learning due to their high dimensionality and absent or questionable feature-level interpretability. This article explores whether large language models (LLMs) could address this by extracting a small number of interpretable features from text. Additionally, we introduce an LLM-based feature discovery mechanism that autonomously identifies key attributes directly from the data, thereby reducing reliance on manual feature selection. 
We demonstrate this process primarily on two datasets (CORD-19 and M17+) containing several thousand scientific articles with a target proxy for research impact. An evaluation based on the statistically significant correlation with research impact has shown that LLama 2-generated features are semantically meaningful, and we consequently used these features for text classification—predicting the binary citation rate for the CORD-19 dataset and an ordinal 5-class expert-awarded grade for the M17+ dataset. Machine-learning models trained on the LLM-generated features provided similar predictive performance to the state-of-the-art embedding model SciBERT for scientific text—achieving competitive results with only 62 features (as opposed to 768 in SciBERT embeddings) that are fully interpretable, capturing qualities such as article methodological rigour, novelty, or grammatical correctness. Consequently, we apply action rule mining to derive a small number of well-interpretable rules, and we further believe that this technique can be beneficial for other white-box methods beyond rule learning.  In addition, to verify the universality of our approach, we applied it to additional datasets: BANKING77, Hate Speech, and Food Hazard (without action rule mining). 

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

### Feature Generation

Feature generation on user-defined features. The corresponding notebook is [ffeature_extraction-CORD19-M17plus.ipynb](notebooks/feauture_extraction/feature_extraction-CORD19-M17plus.ipynb).

### LLM-based feature generation with automatic feature discovery

LLM-based feature discovery with GPT-4o and automated feature generation with GPT-4o-mini. The corresponding notebook with feature discovery prompt is [llm-chatgpt.ipynb](notebooks/feauture_extraction/llm-chatgpt.ipynb).

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
