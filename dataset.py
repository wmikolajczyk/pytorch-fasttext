import logging
from collections import defaultdict

import nltk
import torch
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from tqdm import tqdm

nltk.download("stopwords")
nltk.download("wordnet")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Word2VecDataset(Dataset):
    """Text8 dataset
    https://github.com/RaRe-Technologies/gensim-data
    text8 - First 100,000,000 bytes of plain text from Wikipedia (for testing), no punctuation
    """

    def __init__(self, ngram_len: int, context_size: int):
        import gensim.downloader as api

        raw_dataset = api.load("text8")  # load as Dataset object (iterator)

        en_stop_words = stopwords.words("english")
        lem = nltk.WordNetLemmatizer()
        clean_dataset, word_freqs = self.preprocess_text(
            raw_dataset, en_stop_words, lem
        )
        ngrams = self.prepare_ngrams(list(word_freqs.keys()), ngram_len=ngram_len)
        training_data = self.prepare_training_data(
            clean_dataset, context_size=context_size
        )
        self.data = torch.tensor(training_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """On index 0 - word, on index 1 - context word"""
        x = self.data[index, 0]
        y = self.data[index, 1]
        return x, y

    def preprocess_text(self, raw_dataset, en_stop_words, lem):
        """Preprocess text and gather word frequencies"""
        logger.info("preprocessing text")
        word_freqs = defaultdict(int)

        clean_dataset = []
        for article in tqdm(raw_dataset):
            clean_article = []
            for word in article:
                # remove stopwords
                if word in en_stop_words:
                    continue
                # lemmatize words
                lem_word = lem.lemmatize(word, nltk.corpus.reader.wordnet.VERB)

                word_freqs[lem_word] += 1
                clean_article.append(lem_word)
            clean_dataset.append(clean_article)
        return clean_dataset, word_freqs

    def prepare_ngrams(self, words: list, ngram_len: int):
        """Create dict of ngrams for word
        ngrams + whole word, example: "where": ['<wh', 'whe', 'her', 'ere', 're>', '<where>']
        """
        word_dict = {}
        for word in words:
            ngrams = []
            ngram_word = f"<{word}>"
            for start_idx in range(len(ngram_word) - ngram_len + 1):
                ngram = ngram_word[start_idx : start_idx + ngram_len]
                ngrams.append(ngram)
            ngrams.append(ngram_word)

            word_dict[word] = ngrams
        return word_dict

    def prepare_training_data(self, clean_dataset, context_size):
        """Prepare context word positive and negative pairs"""
        context_pairs = []
        for article in tqdm(clean_dataset):
            for word_idx in range(len(article)):
                min_idx = max(word_idx - context_size, 0)
                max_idx = min(word_idx + context_size + 1, len(article))
                for context_word_idx in range(min_idx, max_idx):
                    # skip current word
                    if context_word_idx == word_idx:
                        continue
                    context_pairs.append([article[word_idx], article[context_word_idx]])
        return context_pairs


Word2VecDataset(ngram_len=3, context_size=2)
