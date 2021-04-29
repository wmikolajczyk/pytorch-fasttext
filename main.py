import logging

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

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
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
model = Word2VecNegSampling(
    vocab_size=len(train_dataset.word_to_idx),
    embedding_size=config["embedding_size"],
)
loss = nn.BCEWithLogitsLoss()
# Adam tends to stuck in local minima - so I used AdamW
optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
losses = []

model.train().to(device)
batch_loss = 0
logger.info("starting training")
for epoch in range(config["num_epochs"]):
    tqdm_bar = tqdm(train_loader)
    for i, batch in enumerate(tqdm_bar):
        #         reshaped_batch = batch.view(
        #             config["batch_size"] * (1 + train_dataset.negative_samples_count),
        #             (1 + train_dataset.negative_samples_count),
        #         ).to(device)
        reshaped_batch = batch.view(
            batch.shape[0] * (1 + train_dataset.negative_samples_count),
            (1 + train_dataset.negative_samples_count),
        ).to(device)
        # clear gradients stored in model parameters
        # not calling this would be required for gradient accumulation
        optimizer.zero_grad()
        # calculate loss
        logits = model(reshaped_batch[:, 0], reshaped_batch[:, 1])
        batch_loss = loss(logits, reshaped_batch[:, 2].float())
        # tqdm set desc - every Nth batch - with batch loss
        # calculate gradients and store them in model parameters - batch_loss.item() - to don't keep the reference
        batch_loss.backward()
        # update model weigths based on gradients in model parameters and learning rate
        optimizer.step()

        losses.append(batch_loss.item())
        if i % 10 == 0:
            tqdm_bar.set_postfix({"batch loss": batch_loss.item()})

# save model
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    config["model_output_path"],
)
