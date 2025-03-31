import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.utils import resample
import re
import json
import os
from nltk import download
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import re
import string
import contractions

download('averaged_perceptron_tagger')
download('stopwords')
download('punkt')


def plot_categorical_distribution(df, target_variable, llm_feature):
    plt.figure(figsize=(10, 6))
    sns.set_palette('viridis')

    if df[df[llm_feature] == 'medium'].shape[0] != 0:
        category_order = ['low', 'medium', 'high']
        ax = sns.countplot(data=df, x=llm_feature, hue=target_variable, order=category_order, palette='viridis')
    else:
        ax = sns.countplot(data=df, x=llm_feature, hue=target_variable, palette='viridis')

    plt.ylabel('Count')
    # if '0' in categorical_variable:
    #     categorical_variable = ford_categories[categorical_variable]
    plt.xlabel(llm_feature)

    plt.title(f'Distribution of {llm_feature} by {target_variable}')

    for p in ax.patches:
        if p.get_height() != 0:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                        textcoords='offset points')

    plt.legend(title=target_variable)
    plt.xticks(rotation=45)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.show()


def get_stat_significance(df, categorical_variable, target_variable):
    cross_tab = pd.crosstab(df[categorical_variable], df[target_variable])
    # print(cross_tab)

    chi2, p_val, _, expected_freq = chi2_contingency(cross_tab)

    # Calculate effect size (Cramér's V)
    n = cross_tab.sum().sum()  # total number of observations
    k = min(cross_tab.shape) - 1  # smaller of (rows - 1) or (columns - 1)
    cramers_v = np.sqrt(chi2 / (n * k))

    print("Chi-Squared Value:", chi2)
    print("P-value:", p_val)
    print("Cramér's V (Effect Size):", cramers_v)

    # Interpret effect size (Cramér's V)
    if cramers_v <= 0.1:
        effect_size_interpretation = "Weak association"
    elif cramers_v <= 0.3:
        effect_size_interpretation = "Moderate association"
    else:
        effect_size_interpretation = "Strong association"

    print("Effect Size:", effect_size_interpretation)

    if p_val < 0.05:
        print(f"There is a significant association between {categorical_variable} and {target_variable}.")
    else:
        print(f"There is no significant association between {categorical_variable} and {target_variable}")

    print("---------------------------------------------------------------")


def get_stat_significance_bootstrap(df, categorical_variable, target_variable, n_bootstrap=2500):
    cross_tab = pd.crosstab(df[categorical_variable], df[target_variable])

    chi2, p_val, _, expected_freq = chi2_contingency(cross_tab)

    # Calculate effect size (Cramér's V)
    n = cross_tab.sum().sum()  # total number of observations
    k = min(cross_tab.shape) - 1  # smaller of (rows - 1) or (columns - 1)
    cramers_v = np.sqrt(chi2 / (n * k))

    # Bootstrap validation
    bootstrap_p_values = []
    i = 0
    for _ in range(n_bootstrap):
        df_sample = resample(df)
        cross_tab_sample = pd.crosstab(df_sample[categorical_variable], df_sample[target_variable])
        if cross_tab_sample.shape == cross_tab.shape:  # Ensure same shape for comparison
            chi2_sample, p_val_sample, _, _ = chi2_contingency(cross_tab_sample)
            if p_val_sample < 0.05:
                i += 1
            bootstrap_p_values.append(p_val_sample)

    mean_p_val = np.mean(bootstrap_p_values)

    # print(f"Chi-Squared Value: {chi2}")
    print(f"Original P-value: {p_val}")
    print(f"Bootstrap Mean P-value: {mean_p_val}")
    print(f"Relationship is significant in {i}/{n_bootstrap} bootstrap samples.")

    print(f"Cramér's V (Effect Size): {cramers_v}")

    # Interpret effect size (Cramér's V)
    effect_size_interpretation = (
        "Weak association" if cramers_v <= 0.1 else
        "Moderate association" if cramers_v <= 0.3 else
        "Strong association"
    )
    print(f"Effect Size: {effect_size_interpretation}")

    if p_val < 0.05:
        print(f"Significant association between {categorical_variable} and {target_variable}.")
    else:
        print(f"No significant association between {categorical_variable} and {target_variable}.")

    print("---------------------------------------------------------------")


def extract_json(response: str):
    """Extract JSON content from a formatted string."""
    match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = response.strip('```json').strip('```')

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Chyba při dekódování JSON: {e}")
        return None


def process_txt_files(folder_path, prefix):
    """Zpracuje všechny txt soubory začínající prefixem (např. 'bank_part') ve složce a vrátí Pandas DataFrame."""
    all_data = []

    # Get files with prefix and end with .txt
    files = [f for f in os.listdir(folder_path) if f.startswith(prefix) and f.endswith(".txt")]

    # Order by number
    files.sort(key=lambda x: int(re.search(r'chunk(\d+)', x).group(1)))

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    json_obj = json.loads(line.strip())
                    content_str = json_obj.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message",
                                                                                                           {}).get(
                        "content", "")
                    extracted_json = extract_json(content_str)

                    if extracted_json:
                        row = {"id": json_obj["id"], "custom_id": json_obj["custom_id"]}
                        for feature in extracted_json.get("features", []):
                            row[feature["feature_name"]] = feature["answer"]

                        all_data.append(row)
                except json.JSONDecodeError:
                    print(f"Chyba dekódování JSON v souboru {filename}")

    df = pd.DataFrame(all_data)
    return df


def expand_contractions(text: str):
    """Expand contractions in the text."""
    return contractions.fix(text)


def split_hyphenated(tokens):
    """Split hyphenated words into individual words."""
    new_tokens = []
    for token in tokens:
        if "-" in token:
            new_tokens.extend(token.split("-"))  # Split into separate words
        else:
            new_tokens.append(token)
    return new_tokens


def remove_possessives(tokens):
    return [re.sub(r"'s\b", "", token) for token in tokens]


def get_wordnet_pos(word):
    """Map POS tag to first character for WordNetLemmatizer."""
    tag = pos_tag([word])[0][1][0].upper()
    return {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}.get(tag, wordnet.NOUN)


def tokenize(text: str):
    """Basic tokenization using regex and NLTK."""
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.replace('/', ' ')
    text = re.sub(r'\b(\w\.){2,}', lambda m: m.group(0).replace('.', ''), text)
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return word_tokenize(text)


def remove_punctuation(tokens):
    """Remove punctuation from tokenized words."""
    return [word for word in tokens if word not in string.punctuation]


def remove_stopwords(tokens: list, stop_words: set):
    """Remove dynamically identified stopwords and non-alphabetic words."""
    return [word for word in tokens if word not in stop_words]


def lemmatize(tokens: list):
    """Lemmatize words using POS tags."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]


def preprocessing(text: str):
    """Full preprocessing pipeline with dynamic stopword removal."""
    text = expand_contractions(text)
    text = text.lower()
    tokens = tokenize(text)
    tokens = remove_possessives(tokens)
    tokens = remove_punctuation(tokens)
    stop_words = set(stopwords.words('english'))
    tokens = split_hyphenated(tokens)
    tokens = remove_stopwords(tokens, stop_words)
    tokens = lemmatize(tokens)
    return " ".join(tokens)
