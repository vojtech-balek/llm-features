# LLM Extraction of Interpretable Features from Text

This repository contains the code and supplementary materials for the scientific article **"LLM Extraction of Interpretable Features from Text."** The aim of this project is to demonstrate how large language models (LLMs) can be used to extract interpretable features from textual data.

## Table of Contents

- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments and Notebooks](#experiments-and-notebooks)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Existing text representations such as embeddings and bag-of-words are not suitable for rule learning due to their high dimensionality and absent or questionable feature-level interpretability. This article explores whether large language models (LLMs) could address this by extracting a small number of interpretable features from text.
We demonstrate this process on two datasets (CORD-19  and M17+) containing several thousand scientific articles from multiple disciplines and a target being a proxy for research impact. An evaluation based on testing for the statistically significant correlation with research impact has shown that LLama 2-generated features are semantically meaningful. We consequently used these generated features in text classification to predict the binary target variable representing the citation rate for the CORD-19 dataset and the ordinal 5-class target representing an expert-awarded grade in the M17+ dataset. Machine-learning models trained on the LLM-generated features provided similar predictive performance to the state-of-the-art embedding model SciBERT for scientific text. Not only did the LLM use only 62 features compared to 768 features in SciBERT embeddings, but these features were fully directly interpretable, as they corresponded to notions such as article methodological rigour, novelty, or grammatical correctness. Consequently, we apply action rule mining resulting in a small number of well-interpretable rules. Consistently competitive results obtained with the same LLM feature set across both thematically diverse datasets show that this approach generalizes across domains.  We also assume this technique could be used not only in rule learning but also in other white-box methods. Our results are replicable due to the use of open LLM. 

## Repository Structure

The repository is organized as follows:

root/
│
├── notebooks/
│ ├── data_preprocessing.ipynb # Preprocessing of the textual data
│ ├── feature_extraction.ipynb # Extraction of features using LLMs
│ ├── feature_interpretation.ipynb # Interpretation of the extracted features
│ ├── model_evaluation.ipynb # Evaluation of the model and extracted features
│ └── visualization.ipynb # Visualization of results and features
│
├── data/
│ ├── raw/ # Raw data used in the experiments
│ └── processed/ # Processed data after cleaning and preparation
│
├── src/
│ ├── utils.py # Utility functions
│ ├── feature_extractor.py # Scripts for feature extraction
│ ├── interpreter.py # Scripts for interpreting extracted features
│ └── evaluate.py # Scripts for evaluating models
│
└── README.md # Project overview and instructions


## Installation

To get started, clone this repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/LLM-feature-extraction.git
cd LLM-feature-extraction
pip install -r requirements.txt


