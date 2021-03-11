import logging
from collections import defaultdict

import nltk
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

    def __init__(self, context_size: int):
        import gensim.downloader as api

        raw_dataset_iter = api.load("text8")  # load as Dataset object (iterator)
        raw_dataset = [
            article for article in raw_dataset_iter
        ]  # iterate through and create list
        self.data = self.preprocess_text(raw_dataset)

    def preprocess_text(self, raw_dataset):
        """Preprocess text and gather word frequencies
        :param raw_dataset:
        :return:
        """
        logger.info("preprocessing text")
        en_stop_words = stopwords.words("english")
        lem = nltk.WordNetLemmatizer()

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        pass


Word2VecDataset(context_size=4)
