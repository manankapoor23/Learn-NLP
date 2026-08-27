# Learn-NLP

A personal learning log for NLP fundamentals, working up from basic text preprocessing (tokenization, stemming, stopword removal) toward classic feature-extraction techniques (Bag of Words, TF-IDF) used in machine learning / deep learning / generative AI pipelines. Everything here is small, self-contained NLTK/scikit-learn experiments rather than a packaged library.

## Notebooks

| Notebook | Covers |
|---|---|
| [`Tokenisation1.ipynb`](Tokenisation1.ipynb) | Sentence and word tokenization with NLTK — `sent_tokenize`, `word_tokenize`, `wordpunct_tokenize`, `TreebankWordTokenizer` |
| [`stopwords.ipynb`](stopwords.ipynb) | Removing English stopwords with NLTK's `stopwords` corpus, combined with sentence tokenization |
| [`porter.ipynb`](porter.ipynb) | Stemming with NLTK's `PorterStemmer` |
| [`lemmatisation.ipynb`](lemmatisation.ipynb) | Lemmatization — placeholder/work in progress, no implementation yet |
| [`postag.ipynb`](postag.ipynb) | Part-of-speech tagging with `nltk.pos_tag`, run after tokenization and stopword removal on a sample paragraph |
| [`named_ER.ipynb`](named_ER.ipynb) | Named entity recognition with `nltk.ne_chunk` over POS-tagged tokens |
| [`TD-IDF/info.txt`](TD-IDF/info.txt) | Notes on the TF-IDF formulas (term frequency, inverse document frequency) — reference notes only, no code yet |
| [`BOW_Implementation/bow_nltk.ipynb`](BOW_Implementation/bow_nltk.ipynb) | Loads the SMS Spam Collection dataset as a first step toward a Bag-of-Words model |
| [`BOW_Implementation/SpamClassifier-master/script.ipynb`](BOW_Implementation/SpamClassifier-master/script.ipynb) | Full Bag-of-Words pipeline on the SMS Spam Collection dataset: regex cleaning, stemming, stopword removal, and vectorization with scikit-learn's `CountVectorizer` (including bigrams) |
| `ai_assgn_1.ipynb` | Empty placeholder — not yet started |

## Tech stack

- Python 3 / Jupyter Notebook
- [NLTK](https://www.nltk.org/) — tokenization, stemming, stopwords, POS tagging, NER
- [scikit-learn](https://scikit-learn.org/) — `CountVectorizer` for Bag of Words
- [pandas](https://pandas.pydata.org/) — loading the SMS Spam Collection dataset

## How to run

```bash
pip install nltk scikit-learn pandas jupyter
jupyter notebook
```

Most notebooks call `nltk.download(...)` inline for the corpora/models they need (`punkt_tab`, `stopwords`, `averaged_perceptron_tagger_eng`, `maxent_ne_chunker_tab`, `words`), so an internet connection is needed the first time each notebook runs.

`BOW_Implementation/bow_nltk.ipynb` and `SpamClassifier-master/script.ipynb` read `smsspamcollection/SMSSpamCollection` via an absolute local path — update the path to match your machine, or run the notebook from the `SpamClassifier-master/` directory.
