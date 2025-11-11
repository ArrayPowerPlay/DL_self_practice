import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import math


class RNNTrainer():
    """Class used for training RNNs models (RNN, LSTM, GRU)"""
    def __init__(self, model, vocab_size, train_loader, val_loader=None, lr=1e-3,
                  num_epochs=10, gradient_clip_val=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab_size = vocab_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.gradient_clip_val = gradient_clip_val
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.train_loss = []
        self.val_loss = []
        self.train_ppl = []
        self.val_ppl = []


    def training_step(self):
        # Implement training in one epoch
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        state = None

        # Add tqdm for training progress
        pbar = tqdm(self.train_loader, desc='Training')

        for X, Y in pbar:
            X, Y = X.to(self.device), Y.to(self.device)
            # Forward pass
            y_hat, state = self.model(X, state)
            # Detach 'state' to interrupt computational graph
            if state is not None:
                if isinstance(state, tuple):    # LSTM
                    state = tuple(s.detach() for s in state)
                else:                           # RNN/GRU
                    state = state.detach()

            loss = self.criterion(y_hat.reshape(-1, self.vocab_size), Y.reshape(-1))
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            if self.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            
            # Update parameters
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss


    def evaluate_step(self):
        # Calculate loss in evaluation set
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        state = None

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Evaluating')

            for X, Y in pbar:
                X, Y = X.T.to(self.device), Y.T.to(self.device)
                # Forward pass
                y_hat, state = self.model(X, state)

                loss = self.criterion(y_hat.reshape(-1, self.vocab_size), Y.reshape(-1))
                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            return avg_loss


    def fit(self):
        # Train model and show progress bar
        for epoch in range(self.num_epochs):
            # Training
            train_loss_epoch = self.training_step()
            self.train_loss.append(train_loss_epoch)
            self.train_ppl.append(math.exp(self.train_loss))
            # Evaluating
            if self.val_loader is not None:
                val_loss_epoch = self.evaluate_step()
                self.val_loss.append(val_loss_epoch)
                self.val_ppl.append(math.exp(self.val_loss))


    def plot(self):
        # Show plot of training and evaluation loss
        plt.figure(figsize=(10, 6))
        epochs = list(range(1, len(self.train_loss) + 1))

        plt.plot(epochs, self.train_loss, label='train_loss')
        plt.plot(epochs, self.val_loss, label='val_loss')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.5)

        plt.xticks(epochs)
        plt.tight_layout()
        plt.show()