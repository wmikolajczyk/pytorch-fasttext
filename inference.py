import logging

import torch
import torch.nn.functional as F

from dataset import Word2VecDataset
from model import Word2VecNegSampling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = {
    "num_epochs": 10,
    "batch_size": 4096,
    "learning_rate": 0.005,
    "embedding_size": 256,
    "model_output_path": "trained_model",
}


device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = Word2VecDataset(context_size=2)
model = Word2VecNegSampling(
    vocab_size=len(train_dataset.word_to_idx), embedding_size=config["embedding_size"]
)

model.load_state_dict(
    torch.load("trained_model", map_location=torch.device("cpu"))["model_state_dict"]
)
model.eval()

with torch.no_grad():
    embeddings = model.embedding(torch.arange(len(train_dataset.word_to_idx)))
    word1 = "plant"
    word2 = "king"
    emb1 = embeddings[train_dataset.word_to_idx[word1]]
    emb2 = embeddings[train_dataset.word_to_idx[word2]]
    print(f"word1: {word1}, word2: {word2}")
    result = torch.dot(emb1, emb2)
    print(f"dot product: {result}")
    norm_emb1 = F.normalize(emb1.unsqueeze(dim=1), p=2)
    norm_emb2 = F.normalize(emb2.unsqueeze(dim=1), p=2)
    euc_dist = torch.cdist(norm_emb1, norm_emb2)
    # print(euc_dist)
    print(f"Euclidean distance sum: {euc_dist.sum()}")
