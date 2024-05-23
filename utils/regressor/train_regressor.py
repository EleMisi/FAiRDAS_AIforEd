import os

import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn

import const_define as cd
from utils.callbacks import EarlyStopping


class Regressor(nn.Module):
    def __init__(self, in_features, hidden_chs):
        super().__init__()
        self.model = torch.nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_chs),
            nn.ReLU(),
            nn.Linear(in_features=hidden_chs, out_features=hidden_chs * 2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_chs * 2, out_features=hidden_chs),
            nn.ReLU(),
            nn.Linear(in_features=hidden_chs, out_features=1)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':

    cd.set_seed(424242)

    # Load Data
    data = pd.read_csv(os.path.join(cd.PROJECT_DIR, 'utils', 'training_data.csv'))
    data = data.drop(['Unnamed: 0'], axis=1)
    # Define features and target tensors
    X = data[[c for c in data.columns if c not in ['score']]]
    y = data['score']
    X = torch.tensor(X.to_numpy(), dtype=torch.float32)
    y = torch.tensor(y.to_numpy(), dtype=torch.float32).reshape(-1, 1)

    # Define model
    hidden_chs = 256
    model = Regressor(X.shape[1], hidden_chs)

    # Define optimizer and loss fn
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss(reduction='mean')

    # Define training parameters
    batch_size = 32
    n_epochs = 100
    batch_start = torch.arange(0, len(data), batch_size)
    early_stopping = EarlyStopping(patience=10, verbose=False, delta=1e-8,
                                   path=os.path.join(cd.PROJECT_DIR, 'utils',
                                                     'regressor_checkpoint.pt'), )
    history = []

    # training loop
    for epoch in range(n_epochs):
        # Shuffle data
        indices = torch.randperm(X.shape[0])
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        history_batch = []
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                optimizer.zero_grad()

                # take a batch
                X_batch = X_shuffled[start:start + batch_size]
                y_batch = y_shuffled[start:start + batch_size]

                assert X_batch.shape[0] == y_batch.shape[0]
                # forward pass
                y_pred = model(X_batch)
                assert y_pred.shape == y_batch.shape

                loss = loss_fn(y_pred, y_batch)
                # backward pass
                loss.backward()
                # update weights
                optimizer.step()
                # Log
                history_batch.append(float(loss))

                optimizer.zero_grad()

            history.append(np.mean(history_batch))
            print()
            print('\tMSE:', history[-1], flush=True)
            print()

            path = early_stopping(np.mean(history_batch), model, optimizer)

            if early_stopping.early_stop:
                print("Early stopping at epoch {}!".format(epoch))
                break
