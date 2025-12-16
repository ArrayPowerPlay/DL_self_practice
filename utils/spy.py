import torch
import math
import collections
from torch import nn
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from IPython.display import clear_output, display
from torch.utils.data import DataLoader, TensorDataset


class Module(nn.Module):
    """Create a base class for model-type class"""
    def __init__(self, lr=None):
        super().__init__()
        self._trainer = None
        self.lr = lr
        self.board = {}
        self._fig = None
        self._ax = None


    def loss(self, y_hat, y):
        raise NotImplementedError
    

    def set_trainer(self, trainer):
        self._trainer = trainer
    

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l
    

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l
    

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)
    

    def plot(self):
        """Plot all training metrics"""
        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(9, 6))
        else:
            self._ax.cla()
        
        # Plot all metrics from board
        for key in self.board.keys():
            if self.board[key]:
                self._ax.plot(range(1, len(self.board[key]) + 1), self.board[key], label=key)
        
        self._ax.set_xlabel('Epoch')
        self._ax.legend(loc='best')
        self._fig.tight_layout()
        
        if clear_output is not None and display is not None:
            clear_output(wait=True)
            display(self._fig)
        else:
            self._fig.canvas.draw_idle()
            plt.pause(0.001)
    

class Classifier(Module):
    """Base class of classification models"""
    def __init__(self, lr=None):
        super().__init__(lr)
        self.board['val_acc'] = []
    

    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute accuracy. Y_hat and Y can have different shapes."""
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        Y = Y.reshape(-1)
        pred = torch.argmax(Y_hat, dim=1)
        compare = (pred == Y).type(torch.float32) 
        return compare.mean() if averaged else compare
    

    def loss(self, Y_hat, Y, averaged=True):
        """Compute cross-entropy loss."""
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        Y = Y.reshape(-1)
        return nn.functional.cross_entropy(Y_hat, Y, reduction='mean' if averaged else 'none')

    
class Trainer:
    def __init__(self, max_epochs, gradient_clip_val=None, device=None):
        self.max_epochs = int(max_epochs)
        self.gradient_clip_val = gradient_clip_val
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    
    def prepare_data(self, data):
        self.train_loader = data.train_dataloader()
        self.val_loader = data.val_dataloader()


    def prepare_model(self, model):
        self.model = model.to(self.device)
        model.set_trainer(self)
        self.optimizer = model.configure_optimizers()


    def _clip_gradients(self):
        if self.gradient_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)


    def fit_epoch(self):
        self.model.train()
        train_total, train_batches = 0.0, 0
        for batch in self.train_loader:
            batch = [b.to(self.device) for b in batch]
            loss = self.model.training_step(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self._clip_gradients()
            self.optimizer.step()
            train_total += loss.item()
            train_batches += 1

        train_loss = train_total / train_batches if train_batches else 0.0

        val_loss = None
        val_acc = None
        if self.val_loader is not None:
            self.model.eval()
            with torch.no_grad():
                val_total, val_batches = 0.0, 0
                acc_total = 0.0
                for batch in self.val_loader:
                    batch = [b.to(self.device) for b in batch]
                    Y_hat = self.model(*batch[:-1])
                    Y = batch[-1]
                    loss = self.model.loss(Y_hat, Y)
                    val_total += loss.item()
                    val_batches += 1
                    
                    # Accumulate accuracy if model has accuracy method
                    if hasattr(self.model, 'accuracy'):
                        acc = self.model.accuracy(Y_hat, Y).item()
                        acc_total += acc
                
                val_loss = val_total / val_batches if val_batches else None
                if hasattr(self.model, 'accuracy') and val_batches > 0:
                    val_acc = acc_total / val_batches

        return train_loss, val_loss, val_acc

        
    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        
        # Initialize board in model for loss tracking
        if 'train_loss' not in model.board:
            model.board['train_loss'] = []
        if 'val_loss' not in model.board:
            model.board['val_loss'] = []
        if 'val_acc' not in model.board:
            model.board['val_acc'] = []

        for epoch in range(self.max_epochs):
            train_loss, val_loss, val_acc = self.fit_epoch()
            model.board['train_loss'].append(train_loss)
            if val_loss is not None:
                model.board['val_loss'].append(val_loss)
            if val_acc is not None:
                model.board['val_acc'].append(val_acc)
            model.plot()
        if model._fig is not None:
            plt.close(model._fig)


class DataModule:
    """Implement basic functions for data manipulation which supports training model"""
    def get_dataloader(self, train):
        raise NotImplementedError
    

    def train_dataloader(self):
        return self.get_dataloader(train=True)
    

    def val_dataloader(self):
        return self.get_dataloader(train=False)
    

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=train)


class FashionMNIST(DataModule):
    """FashionMNIST dataset from torchvision"""
    def __init__(self, batch_size=64, resize=(28, 28), root='../data/fashion-mnist'):
        super().__init__()
        self.batch_size = batch_size
        self.resize = resize
        self.root = root
        self.num_workers = 2
        self.mean = [0.0]  # Grayscale image, no normalization
        self.std = [1.0]
        
        # Định nghĩa transform
        trans = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])
        
        # Download + load dataset
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)
    

    def text_labels(self, indices):
        """Return text labels for indices"""
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        if isinstance(indices, int):
            return labels[indices]
        return [labels[int(i)] for i in indices]
    

    def get_dataloader(self, train):
        """Get dataloader for training or validation"""
        data = self.train if train else self.val
        return DataLoader(data, self.batch_size, shuffle=train, num_workers=self.num_workers)


class CIFAR10(DataModule):
    """CIFAR-10 dataset from torchvision"""
    def __init__(self, batch_size=64, resize=(32, 32), root='../data/cifar-10'):
        super().__init__()
        self.batch_size = batch_size
        self.root = root
        self.num_workers = 2
        self.mean = [0.491, 0.482, 0.447]
        self.std = [0.247, 0.243, 0.262]
        
        # Define transform with data augmentation for training
        train_trans = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(resize[0], padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, 
                               std=self.std)
        ])
        
        test_trans = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean,
                               std=self.std)
        ])
        
        # Download + load dataset
        self.train = torchvision.datasets.CIFAR10(
            root=self.root, train=True, transform=train_trans, download=True)
        self.val = torchvision.datasets.CIFAR10(
            root=self.root, train=False, transform=test_trans, download=True)
    

    def text_labels(self, indices):
        """Return text labels for indices (accepts single int or list)"""
        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']
        if isinstance(indices, int):
            return labels[indices]
        return [labels[int(i)] for i in indices]
    

    def get_dataloader(self, train):
        """Get dataloader for training or validation"""
        data = self.train if train else self.val
        return DataLoader(data, self.batch_size, shuffle=train, num_workers=self.num_workers)


def visualize_prediction(model, data, num_examples=8, trainer=None):
    """Function for visualizing predictions if images for image classification tasks"""
    if not hasattr(data, 'text_labels'):
        raise ValueError("Dataset must support 'text_labels' method")
    
    model.eval()
    num_cols = 4
    num_rows = math.ceil(num_examples / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3*num_rows))

    axes = axes.flatten()
    device = trainer.device if trainer else 'cpu'
    cnt = 0

    with torch.no_grad():
        for batch in data.val_dataloader():
            batch = [b.to(device) for b in batch]
            X, Y = batch
            Y_hat = model(X)

            for i in range(len(X)):
                if cnt >= num_examples:
                    break

                img = X[i].cpu()
                true_label = Y[i].item()
                pred_label = Y_hat[i].argmax().item()

                # Denormalize if the data class has normalization stats
                if hasattr(data, 'mean') and hasattr(data, 'std'):
                    mean = torch.tensor(data.mean).view(-1, 1, 1)
                    std = torch.tensor(data.std).view(-1, 1, 1)
                    img = img * std + mean

                if img.shape[0] == 1:    # gray image
                    axes[cnt].imshow(img.squeeze().numpy())
                else:
                    axes[cnt].imshow(img.permute(1, 2, 0).numpy())

                true_text = data.text_labels(true_label)
                pred_text = data.text_labels(pred_label)

                color = 'green' if true_label == pred_label else 'red'
                axes[cnt].set_title(f"True: {true_text}\nPred: {pred_text}",
                                    color=color, fontsize=10)
                axes[cnt].axis('off')
                cnt += 1
            
            if cnt >= num_examples:
                break

    # Turn off axis of redundant axes
    for idx in range(cnt, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


class Vocab:
    """Build vocabulary for language models and other convenient functions"""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten 2D list if necessary
        if tokens and isinstance(tokens[0], list):
            tokens = [word for line in tokens for word in line]
        # Count words frequency
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x : x[1], reverse=True)

        # convert token to index and index to token
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + 
                                [token for token, freq in counter.items() if freq >= min_freq])))
        self.token_to_idx = {token : index for index, token in enumerate(self.idx_to_token)}


    def __len__(self):
        return len(self.idx_to_token)
    

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    

    def to_tokens(self, index):
        if hasattr(index, '__len__') and len(index) > 1:
            return [self.idx_to_token[idx] for idx in index]
        return self.idx_to_token[index]
        

    @property
    def unk(self):
        return self.token_to_idx['<unk>']
    

class MachineTranslation(DataModule):
    """This class preprocesses the dataset retrieved from 'https://www.manythings.org/anki/'
    to formats that can be fed into DL models supporting machine translation tasks"""
    # Format of the raw data: English + TAB + The Other Language + TAB + Attribution
    def __init__(self, path, batch_size, num_steps=9, num_train=512, num_val=128):
        super().__init__()
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_train = num_train
        self.num_val = num_val
        self.path = path
        self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(self.load_dataset())


    def load_dataset(self):
        with open(self.path, encoding='utf-8') as f:
            return f.read()


    def _preprocess(self, text):
        # Replace non-breaking space with space
        text = text.replace('\u202f', ' ').replace('\xa0', ' ')
        # Insert space between words and punctuation marks
        no_space = lambda char, prev_char: char in '.,!?' and prev_char != ' '
        out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
                for i, char in enumerate(text.lower())]
        
        return ''.join(out)
    

    def _tokenize(self, text, max_examples=None):
        """This method tokenizes the first 'max_examples' pairs of text sequence, where each
        token if either a word or a punctuation mark"""
        # We append the special '<eos>' token to the end of every sequence to indicate the
        # end of the sequence

        src, tgt = [], []
        for i, line in enumerate(text.split('\n')):
            if max_examples and i > max_examples: break
            parts = line.split('\t')
            if len(parts) >= 2:
                src_sentence = parts[0]
                tgt_sentence = parts[1]

                src.append([t for t in f"{src_sentence} <eos>".split(" ") if t])
                tgt.append([t for t in f"{tgt_sentence} <eos>".split(" ") if t])
                
        return src, tgt
    
        
    def histogram_tokens_per_seq(self, legend, xlabel, ylabel, xlist, ylist):
        """Plot the histogram of number of tokens per text sequence"""
        plt.figure(figsize=(9, 6))

        data = [[len(l) for l in xlist], [len(l) for l in ylist]]
        n, bins, patches = plt.hist(data, label=legend)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        for patch in patches[1]:
            patch.set_hatch('/')
        
        plt.legend()
        

    def _build_arrays(self, raw_text, src_vocab=None, tgt_vocab=None):
        """Build input/label for encoder/decoder"""
        def _build_arrays(sentences, vocab, is_tgt=False):
            pad_or_trim = lambda seq, t: (
                seq[:t] if len(seq) > t else seq + ['<pad>'] * (t - len(seq))
            )

            sentences = [pad_or_trim(s, self.num_steps) for s in sentences]
            if is_tgt:
                sentences = [['<bos>'] + s for s in sentences]
            if vocab is None:
                vocab = Vocab(sentences, min_freq=2)

            array = torch.tensor([vocab[s] for s in sentences])
            valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
            return array, vocab, valid_len
        
        src, tgt = self._tokenize(self._preprocess(raw_text), 
                                  self.num_train + self.num_val)
        src_array, src_vocab, src_valid_len = _build_arrays(src, src_vocab)
        tgt_array, tgt_vocab, _ = _build_arrays(tgt, tgt_vocab, True)
        
        return ((src_array, tgt_array[:, :-1], src_valid_len, tgt_array[:, 1:]), 
                src_vocab, tgt_vocab)
    

    def build(self, src_sentences, tgt_sentences):
        raw_text = '\n'.join([src + '\t' + tgt for src, tgt in 
                              zip(src_sentences, tgt_sentences)])
        
        arrays, _, _ = self._build_arrays(raw_text, self.src_vocab, self.tgt_vocab)
        return arrays


    def get_dataloader(self, train):
        idx = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader(self.arrays, train, idx)
    

class Encoder(nn.Module):
    """Base class interface for the encoder architecture"""
    def forward(self, X, *arg):
        raise NotImplementedError


class Decoder(nn.Module):   
    """Base class interface for the decoder architecture"""
    def init_state(self, enc_all_outputs, *arg):
        raise NotImplementedError
    

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(Module):
    """The base class for the encoder-decoder architecture"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, enc_X, dec_X, *arg):
        enc_all_outputs = self.encoder(enc_X, *arg)
        dec_state = self.decoder.init_state(enc_all_outputs, *arg)
        # Return decoder output only
        return self.decoder(dec_X, dec_state)[0]
    

    def predict_step(self, batch, device, num_steps, save_attention_weights=False):
        """Predict token in evaluation step"""
        batch = [a.to(device) for a in batch]
        # src_valid_len corresponds to the valid lengths for the source sequences
        src, tgt, src_valid_len, _ = batch
        enc_all_outputs = self.encoder(src, src_valid_len)
        dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
        outputs, attention_weights = [tgt[:, 0].unsqueeze(1), ], []

        for _ in range(num_steps):
            Y, dec_state = self.decoder(outputs[-1], dec_state)
            outputs.append(Y.argmax(2))
            # Save attention weights 
            if save_attention_weights:
                attention_weights.append(self.decoder.attention_weights)

        return torch.cat(outputs[1:], 1), attention_weights
    

def bleu(pred_seq, label_seq, k):
    """Compute the BLEU score for evaluating the predicted sequence"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))

    for n in range(1, min(k, len_pred) + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                # Clipping
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def check_shape(a, shape):
    """Check whether the shape of a tensor is equivalent to 'shape'"""
    assert a.shape == shape, f'tensor\'s shape {a.shape} != expected shape {shape}'


def init_seq2seq(module):
    """Applying parameters initialization using xavier uniform for seq2seq model"""
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])


class Seq2SeqEncoder(Encoder):
    """Applying encoder for machine translation tasks using GRU with multiple layers"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)
        self.apply(init_seq2seq)

    
    def forward(self, X, *arg):
        # X shape: (batch_size, num_steps)
        embs = self.embedding(X.T.type(torch.int64))
        # embs shape: (num_steps, batch_size, embed_size)
        outputs, state = self.rnn(embs)
        # output shape: (num_steps, batch_size, num_hiddens)
        # state shape: (num_layers, batch_size, num_hiddens)
        return outputs, state
    

class Seq2SeqDecoder(Decoder):
    """Applying decoder for machine translation tasks using GRU with multiple layers"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.rnn = nn.GRU(num_hiddens + embed_size, num_hiddens, num_layers, dropout=dropout)
        # Use a fully connected layer to predict the probability distribution of the output token
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(init_seq2seq)


    def init_state(self, enc_all_outputs, *arg):
        return enc_all_outputs
    

    def forward(self, X, state):
        # X shape: (batch_size, num_steps)
        # embs shape: (num_steps, batch_size, num_hiddens)
        embs = self.embedding(X.T.type(torch.int64))
        # enc_output shape: (num_steps, batch_size, num_hiddens)
        # hidden_state shape: (num_layers, batch_size, num_hiddens)
        enc_output, hidden_state = state
        # context shape: (batch_size, num_hiddens)
        context = enc_output[-1]
        # Repeat context to align with the decoder input time dimension (num_steps for decoder)
        context = context.repeat(embs.shape[0], 1, 1)
        embs_and_context = torch.cat((embs, context), -1)
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state)
        outputs = self.dense(outputs).swapaxes(0, 1)
        # outputs shape: (batch_size, num_steps, vocab_size)
        # hidden_state shape: (num_layers, batch_size, num_hiddens)
        return outputs, [enc_output, hidden_state]
    

class Seq2Seq(EncoderDecoder):
    """Apply sequence to sequence pattern in machine translation problem"""
    def __init__(self, encoder, decoder, tgt_pad, lr):
        super().__init__(encoder, decoder)
        self.tgt_pad = tgt_pad
        self.lr = lr


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

    def loss(self, Y_hat, Y):
        loss_fn = nn.CrossEntropyLoss()
        vocab_size = Y_hat.shape[-1]
        y_hat = Y_hat.reshape(-1, vocab_size)
        y = Y.reshape(-1)
        l = loss_fn(y_hat, y)
        mask = (y != self.tgt_pad).type(torch.float32)
        return (l * mask).sum() / mask.sum()


class RNNTrainer(Trainer):
    """Trainer specialized for RNN/LSTM/GRU language models"""
    def __init__(self, model, vocab_size, train_loader, val_loader=None, lr=1e-3,
                 num_epochs=10, gradient_clip_val=1.0, device=None):
        super().__init__(max_epochs=num_epochs, gradient_clip_val=gradient_clip_val, device=device)
        self.vocab_size = vocab_size
        self.lr = lr
        self.grad_clip = gradient_clip_val
        # Track cross-entropy losses
        self.train_loss, self.val_loss = [], []
        self.criterion = nn.CrossEntropyLoss()
        self.prepare_data(train_loader, val_loader)
        self.prepare_model(model)


    def fit_epoch(self):
        self.model.train()
        total_loss, num_batches, state = 0.0, 0, None

        for X, Y in tqdm(self.train_loader, desc='Training'):
            X, Y = X.T.to(self.device), Y.T.to(self.device)
            y_hat, state = self.model(X, state)

            if state is not None:
                if isinstance(state, tuple):  # LSTM returns (h, c)
                    state = tuple(s.detach() for s in state)
                else:
                    state = state.detach()

            loss = self.criterion(y_hat.reshape(-1, self.vocab_size), Y.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self._clip_gradients()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches else 0.0
        self.train_loss.append(avg_loss)
    
        if self.val_loader is not None:
            self.model.eval()
            total_loss, num_batches, state = 0.0, 0, None
            with torch.no_grad():
                for X, Y in tqdm(self.val_loader, desc='Evaluating'):
                    X, Y = X.T.to(self.device), Y.T.to(self.device)
                    y_hat, state = self.model(X, state)
                    loss = self.criterion(y_hat.reshape(-1, self.vocab_size), Y.reshape(-1))
                    total_loss += loss.item()
                    num_batches += 1

                avg_loss = total_loss / num_batches if num_batches else 0.0
                self.val_loss.append(avg_loss)


    def fit(self):
        for _ in range(self.max_epochs):
            self.fit_epoch()
        self.plot()


def masked_softmax(X, valid_lens):
    """Softmax with masking for sequence to sequence tasks (utility functions for attention)"""
    # X: 3D shape
    def _sequence_mask(X, valid_len, value=0):
        """Replace elements of X whose length < 'valid_len' to 'value'"""
        # X: 2D shape
        # valid_len: 1D shape
        maxlen = X.shape[1]
        mask = torch.arange(maxlen, dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X
    
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, -1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
    

class DotProductAttention(nn.Module):
    """Scaled dot product attention"""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)

        return torch.bmm(self.dropout(self.attention_weights), values)


class AdditiveAttention(nn.Module):
    """Implement additive attention"""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)


    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(values)
        # Shape of queries and keys after the linear layer:
        # queries: (batch_size, no_of_queries, num_hiddens)
        # keys: (batch_size, no_of_key-value_pairs, num_hiddens)
        feature = queries.unsqueeze(2) + keys.unsqueeze(1)
        feature = torch.tanh(feature)
        # There is only one output of self.w_v, so we remove the
        # last dimension entry from the shape
        scores = self.w_v(feature).squeeze(-1)
        # score shape: (batch_size, no_of_queries, no_of_key-value_pairs)
        # value shape: (batch_size, no_of_key-value_pairs, value dimension)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class AttentionDecoder(Decoder):
    """The base attention-based decoder interfaces"""
    def __init__(self):
        super().__init__()

    
    @property
    def attention_weights(self):
        raise NotImplementedError
    

class Seq2SeqAttentionDecoder(AttentionDecoder):
    """The RNN decoder with attention mechanism"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.attention = AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(init_seq2seq)


    def init_state(self, enc_outputs, enc_valid_lens):
        # outputs shape: (num_steps, batch_size, num_hiddens)
        # hidden_state shape: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)
    

    def forward(self, X, state):
        # enc_outputs shape: (batch_size, num_steps, num_hiddens)
        # hidden_state shape: (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # X shape: (batch_size, num_steps)
        X = self.embedding(X).permute(1, 0, 2)
        # X shape after the embedding layer: (num_steps, batch_size, num_hiddens)
        outputs, self._attention_weights = [], []

        for x in X:
            # query shape: (batch_size, 1, num_hiddens)
            # hidden_state[-1] is the last layer of the previous timestep
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # context shape: (batch_size, 1, num_hiddens)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # Reshape x as: (1, batch_size, embed_size + num_hiddens) to feed into the GRU model
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)

            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]
    

    @property
    def attention_weights(self):
        return self._attention_weights


class MultiHeadAttention(Module):
    """Implement multi-head attention"""
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.num_hiddens = num_hiddens
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)


    def forward(self, queries, keys, values, valid_lens):
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        # valid_lens shape: (batch_size,) or (batch_size, no_of_queries)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        
        output = self.attention(queries, keys, values, valid_lens)
        output_concat = self.transpose_output(output)

        return self.W_o(output_concat)


    def transpose_qkv(self, X):
        """Transposition for parallel computation of multiple attention heads"""
        # X shape: (batch_size, no_of_queries or no_of_key-value_pairs, num_hiddens) 
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        # X shape after reshape: 
        # (batch_size, no_of_queries or no_of_key-value_pairs, num_heads, num_hiddens/num_heads)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[-1])
        # X shape after return:
        # batch_size*num_heads, no_of_queries or no_of_key-value_pairs, num_hiddens/num_heads)


    def transpose_output(self, X):
        """Reverse the operation of 'transpose_qkv'"""
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[-1])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)


class PositionalEncoding(nn.Module):
    """Implementation of positional encoding"""
    # 'max_len' prepare a sufficiently long "repository" of positional encoding for 'num_steps'
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)


    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
    

class PositionWiseFFN(nn.Module):
    """The positionwise feed forward network"""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)


    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
        

class AddNorm(nn.Module):
    """The residual connection followed by layer normalization"""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)


    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class TransformerEncoderBlock(nn.Module):
    """The Transformer encoder block"""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias=False):
        super().__init__()
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout, bias=use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)


    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
    

class TransformerEncoder(Encoder):
    """Implement the Transformer encoder for seq2seq tasks"""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.num_heads = num_heads
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block" + str(i), TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias
            ))


    def forward(self, X, valid_lens):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights

        return X
        # output of a layer of a Transformer encoder: (batch_size, num_steps, num_hiddens)


class TransformerDecoderBlock(nn.Module):
    """The i_th block in the Transformer decoder architecture"""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.attention2 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens, dropout)


    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            # Concat in 'num_steps' dimension
            key_values = torch.cat((state[2][self.i], X), dim=1)

        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)  
        Z = self.addnorm2(Y, Y2)
        Z2 = self.ffn(Z)
        return self.addnorm3(Z, Z2), state
    

class TransformerDecoder(AttentionDecoder):
    """Implement the Transformer decoder for seq2seq tasks"""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_heads = num_heads
        self.num_blks = num_blks
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        
        for i in range(num_blks):
            self.blks.add_module("block" + str(i), TransformerDecoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.LazyLinear(vocab_size)


    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]


    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # Decoder self-attention weights
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # Encoder-decoder attention weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights

        return self.dense(X), state
    

    @property
    def attention_weights(self):
        return self._attention_weights


class PatchEmbedding(nn.Module):
    """Implement patch embedding in vision Transformer"""
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size, stride=patch_size)

    
    def forward(self, X):
        # X shape before conv: (batch_size, num_channels, height, width)
        # X shape after conv: (batch_size, num_hiddens, height//patch_size, width//patch_size)
        # X shape after flatten: (batch_size, num_hiddens, num_patches)
        return self.conv(X).flatten(2).transpose(1, 2)
        # output shape: (batch_size, num_patches, num_hiddens)


class ViTMLP(nn.Module):
    """MLP of the vision Transformer"""
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, X):
        X1 = self.dropout1(self.gelu(self.dense1(X)))
        return self.dropout2(self.dense2(X1))
    

class ViTBlock(nn.Module):
    """Layer implementation of vision Transformer"""
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens, num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout, bias=use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)


    def forward(self, X, valid_lens=None):
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        return X + self.mlp(self.ln2(X))


class ViT(Classifier):
    """Implementation of vision Transformer"""
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads, num_blks,
                 emb_dropout, blk_dropout, lr=0.1, use_bias=False, num_classes=10):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1
        # Positional embedding are learnable
        self.pos_embedding = nn.Parameter(torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        self.lr = lr
        
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(num_hiddens, num_hiddens, mlp_num_hiddens,
                                                  num_heads, blk_dropout, use_bias))
        
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens), nn.Linear(num_hiddens, num_classes))


    def forward(self, X):
        X = self.patch_embedding(X)
        X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), dim=1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])
