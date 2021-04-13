import logging
import random
from collections import defaultdict

import nltk
import torch
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from tqdm import tqdm

nltk.download("stopwords")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random.seed(93)  # for reproducibility

"""
TODO:
- remove dataset limiting
- preprocess dataset
- allow loading preprocessed dataset!
- add negative sampling to getitem method (to select different negative examples each time for index 0)
- refactor code
- check if model is training


- support Ngrams
- calculate ngram frequencies
- cut top N ngrams
"""


# https://github.com/n0obcoder/Skip-Gram-Model-PyTorch
# https://www.kaggle.com/karthur10/skip-gram-implementation-with-pytorch-step-by-step
# https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb


class Word2VecDataset(Dataset):
    """Text8 dataset
    https://github.com/RaRe-Technologies/gensim-data
    text8 - First 100,000,000 bytes of plain text from Wikipedia (for testing), no punctuation

    dataset with transformation: https://git.egnyte-internal.com/datalab/doc-classifier-bert/-/blob/master/boilerplate/datasets.py
    """

    def __init__(self, context_size: int):
        import gensim.downloader as api

        raw_dataset = api.load("text8")  # load as Dataset object (iterator)
        # limit dataset to speed up development - TODO - remove limiting
        raw_dataset = [x for x in raw_dataset][:100]

        en_stop_words = stopwords.words("english")
        clean_dataset, word_freqs = self.preprocess_text(raw_dataset, en_stop_words)
        word_to_idx = {w: i for i, w in enumerate(word_freqs)}

        # word_ngrams = self.prepare_word_ngrams(
        #     list(word_freqs.keys()), ngram_len=ngram_len
        # )
        # ngrams_occurences = self.count_ngrams_occurences(word_freqs, word_ngrams)
        # top_ngrams = ngrams_occurences[:top_ngrams]
        training_data = self.prepare_training_data(
            clean_dataset,
            word_to_idx,
            context_size=context_size,
            negative_samples_count=2,
        )

        self.word_freqs = word_freqs
        self.data = torch.tensor(training_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        [
            [central_word, pos_word, 1],
            [central_word, neg_word1, 0],
            [central_word, neg_word2, 0]
        ]
        """
        # ['<wh', 'whe', 'her', 'ere', 're>', '<where>'] -> [0, 23, 51, 25, 23] (indices of ngrams)
        # ['<wh', 'whe', 'her', 'ere', 're>', '<where>']
        # tensor of long values - indices of ngrams

        central_word = self.data[index, 0]
        context_word = self.data[index, 1]
        is_positive = self.data[index, 2]

        return central_word, context_word, is_positive

    def preprocess_text(self, raw_dataset, en_stop_words):
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

                word_freqs[word] += 1
                clean_article.append(word)
            clean_dataset.append(clean_article)
        return clean_dataset, word_freqs

    def prepare_ngram(self, word: str, ngram_len: int):
        """Prepare list of ngrams for a word"""
        ngrams = []
        word_ngram = f"<{word}>"
        for start_idx in range(len(word_ngram) - ngram_len + 1):
            ngram = word_ngram[start_idx : start_idx + ngram_len]
            ngrams.append(ngram)
        ngrams.append(word_ngram)
        return ngrams

    def prepare_word_ngrams(self, words: list, ngram_len: int):
        """Create dict of ngrams for word"""
        word_ngrams = {}
        for word in words:
            word_ngrams[word] = self.prepare_ngram(word, ngram_len)
        return word_ngrams

    def count_ngrams_occurences(
        self,
        word_freqs: dict[str, int],
        word_ngrams: dict[str, list],
        sort: bool = True,
    ):
        ngram_occurences = defaultdict(int)
        for word, freq in word_freqs.items():
            for ngram in word_ngrams[word]:
                ngram_occurences[ngram] += freq

        if sort:
            return dict(
                sorted(
                    ngram_occurences.items(),
                    key=lambda key_val: key_val[1],
                    reverse=True,
                )
            )
        return ngram_occurences

    def prepare_training_data(
        self,
        clean_dataset: list[list[str]],
        word_to_idx: dict[str, int],
        context_size: int,
        negative_samples_count: int,
    ):
        """Prepare context word positive and negative pairs"""
        # consists of [word, context_word, is_context] where is_context is 0 or 1
        context_pairs = []

        words_list = list(word_to_idx.keys())
        for article in tqdm(clean_dataset):
            for word_idx in range(len(article)):
                min_idx = max(word_idx - context_size, 0)
                max_idx = min(word_idx + context_size + 1, len(article))
                for context_word_idx in range(min_idx, max_idx):
                    # skip current word
                    if context_word_idx == word_idx:
                        continue
                    context_pairs.append(
                        [
                            word_to_idx[article[word_idx]],
                            word_to_idx[article[context_word_idx]],
                            1,
                        ]
                    )
                    # add N negative pairs - each call - negative samples for idx 0 - should return different values
                    #   for now keep this code here - if we want to change it, change seed here
                    for i in range(negative_samples_count):
                        random_word = random.choice(words_list)
                        context_pairs.append(
                            [
                                word_to_idx[article[word_idx]],
                                word_to_idx[random_word],
                                0,
                            ]
                        )
        return context_pairs


# training loop with negative examples
# 1. implement skip gram model
# 2. change skip gram to ngrams -> fasttext
