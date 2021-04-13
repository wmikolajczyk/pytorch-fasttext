import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Word2VecDataset
from model import Word2VecNegSampling

config = {
    "num_epochs": 10,
    "batch_size": 64,
    "learning_rate": 0.005,
    "embedding_size": 512,
    "model_output_path": "trained_model",
}

if __name__ == "__main__":
    train_dataset = Word2VecDataset(context_size=2)
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    model = Word2VecNegSampling(
        vocab_size=len(train_dataset.word_freqs),
        embedding_size=config["embedding_size"],
    )
    loss = nn.BCEWithLogitsLoss()
    # Adam tends to stuck in local minima - so I used AdamW
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    losses = []

    model.train()
    for epoch in tqdm(range(config["num_epochs"])):
        for batch_idx, (
            central_word_batch,
            context_word_batch,
            is_positive_batch,
        ) in enumerate(
            train_loader
        ):  # tqdm + batch_loss as description
            # clear gradients stored in model parameters
            # not calling this would be required for gradient accumulation
            optimizer.zero_grad()
            # calculate loss
            logits = model(central_word_batch, context_word_batch)
            batch_loss = loss(logits, is_positive_batch.float())
            print(batch_loss.item())
            # calculate gradients and store them in model parameters
            batch_loss.backward()
            # update model weigths based on gradients in model parameters and learning rate
            optimizer.step()

            losses.append(batch_loss.item())

    # save model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        config["model_output_path"],
    )
