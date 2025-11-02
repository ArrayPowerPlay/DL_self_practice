import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from tqdm import tqdm


class RNNTrainer():
    """Class used for training RNNs models (RNN, LSTM, GRU)"""
    def __init__(self, model, train_loader, test_loader, vocab_size, lr=1e-3,
                  num_epochs=10, gradient_clip_val=None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.vocab_size = vocab_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.gradient_clip_val = gradient_clip_val
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.train_loss = []
        self.test_loss = []
        self.test_acc = []


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


    def evaluate(self):
        # Calculate loss in test set
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        state = None

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Evaluating')

            for X, Y in self.test_loader:
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
            # Testing
            test_loss_epoch = self.evaluate()
            self.test_loss.append(test_loss_epoch)


    def plot(self):
        # Show plot of training and testing loss
        plt.figure(figsize=(10, 6))
        epochs = list(range(1, len(self.train_loss) + 1))

        plt.plot(epochs, self.train_loss, label='train_loss')
        plt.plot(epochs, self.test_loss, label='test_loss')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.5)

        plt.xticks(epochs)
        plt.tight_layout()
        plt.show()