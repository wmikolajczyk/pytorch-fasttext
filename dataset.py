import logging
import os
import random
from collections import defaultdict
from typing import Dict, List

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
- support Ngrams (change skip gram to ngrams -> fasttext)
- calculate ngram frequencies
- cut top N ngrams

# https://github.com/n0obcoder/Skip-Gram-Model-PyTorch
# https://www.kaggle.com/karthur10/skip-gram-implementation-with-pytorch-step-by-step
# https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb
"""


class Word2VecDataset(Dataset):
    """Text8 dataset
    https://github.com/RaRe-Technologies/gensim-data
    text8 - First 100,000,000 bytes of plain text from Wikipedia (for testing), no punctuation

    dataset with transformation: https://git.egnyte-internal.com/datalab/doc-classifier-bert/-/blob/master/boilerplate/datasets.py
    """

    def __init__(self, context_size: int, negative_samples_count: int = 2):
        import gensim.downloader as api

        raw_dataset = api.load("text8")  # load as Dataset object (iterator)
        # limit dataset to speed up development
        raw_dataset = [x for x in raw_dataset]

        en_stop_words = stopwords.words("english")
        clean_dataset, word_freqs = self.preprocess_text(raw_dataset, en_stop_words)
        ##
        words_limit = 20000
        top_words = [
            el[0]
            for el in sorted(
                word_freqs.items(), key=lambda key_val: key_val[1], reverse=True
            )[:words_limit]
        ]
        word_to_idx = {w: i for i, w in enumerate(top_words)}
        word_to_idx["<UNKNOWN>"] = len(word_to_idx)

        # word_ngrams = self.prepare_word_ngrams(
        #     list(word_freqs.keys()), ngram_len=ngram_len
        # )
        # ngrams_occurences = self.count_ngrams_occurences(word_freqs, word_ngrams)
        # top_ngrams = ngrams_occurences[:top_ngrams]
        training_data_path = "training_data"
        if not os.path.exists(training_data_path):
            logger.info(f"no {training_data_path}")
            training_data = self.prepare_training_data(
                clean_dataset,
                word_to_idx,
                context_size=context_size,
            )
            training_data = torch.tensor(training_data)
            torch.save(training_data, training_data_path)
        else:
            logger.info(f"loading training data from {training_data_path}")
            training_data = torch.load(training_data_path)

        self.word_to_idx = word_to_idx
        self.data = training_data
        self.negative_samples_count = negative_samples_count

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns one positive pair (central and context word)
        and <negative_samples_count> negative pairs
        [
            [central_word, pos_word, 1],
            [central_word, neg_word1, 0],
            [central_word, neg_word2, 0]
        ]
        """
        # ['<wh', 'whe', 'her', 'ere', 're>', '<where>'] -> [0, 23, 51, 25, 23] (indices of ngrams)
        # ['<wh', 'whe', 'her', 'ere', 're>', '<where>']
        # tensor of long values - indices of ngrams

        # central word on index 0 and context word on index 1
        central_word = self.data[index, 0]
        context_word = self.data[index, 1]
        # list of word indexes to sample negative examples from
        # non_context_word_indexes = [
        #     word_index
        #     for word_index in self.word_to_idx.values()
        #     if word_index not in [central_word, context_word]
        # ]
        # word_indexes = list(self.word_to_idx.values())
        max_word_index = len(self.word_to_idx)
        random_words = []
        sampled_word_index = random.randint(0, max_word_index - 1)
        while len(random_words) < self.negative_samples_count:
            # sets are implemented as hash tables, O(1) lookup
            if sampled_word_index not in {central_word, context_word}:
                random_words.append(sampled_word_index)
            sampled_word_index = random.randint(0, max_word_index - 1)

        # add positive pair
        data = [[central_word, context_word, 1]]
        # add negative pairs
        for word_idx in random_words:
            data.append([central_word, word_idx, 0])

        return torch.tensor(data, dtype=torch.long)

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
        word_freqs: Dict[str, int],
        word_ngrams: Dict[str, list],
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
        clean_dataset: List[List[str]],
        word_to_idx: Dict[str, int],
        context_size: int,
    ):
        """Prepare context word positive and negative pairs"""
        # consists of positive examples [word, context_word]
        context_pairs = []

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
                            word_to_idx.get(
                                article[word_idx], word_to_idx["<UNKNOWN>"]
                            ),
                            word_to_idx.get(
                                article[context_word_idx], word_to_idx["<UNKNOWN>"]
                            ),
                        ]
                    )
        return context_pairs
