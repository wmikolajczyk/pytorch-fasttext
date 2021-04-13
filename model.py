import torch
import torch.nn as nn


# TODO:
# skip-gram model
#   objective - maximalize log-likelihood
#   probability of context word - softmax
#   -> independent binary classification class - predict presence or absence of context words
#   for word at position t - all context words = positive, sample negatives random from dictionary
#   negative log likelihood
# n-grams support

# fasttext vs skip-grams - basically only n-grams makes difference -> data processing

# https://medium.datadriveninvestor.com/word2vec-skip-gram-model-explained-383fa6ddc4ae
# https://github.com/n0obcoder/Skip-Gram-Model-PyTorch


class Word2VecNegSampling(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int):
        super().__init__()
        # model architecture
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_size
        )

    def forward(self, central_word, context_word):
        # forward logic
        central_embedding = self.embedding(central_word)
        context_embedding = self.embedding(context_word)
        # this is for positive example
        embedding_product = torch.mul(central_embedding, context_embedding)
        embedding_product = torch.sum(embedding_product, dim=1)
        return embedding_product
