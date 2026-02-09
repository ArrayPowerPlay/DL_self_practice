import torch
import os
import re
import math
import collections
from torch import nn
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
import datasets
from datasets import load_dataset
import torchvision
from torchvision import transforms
from IPython.display import clear_output, display
from torch.utils.data import DataLoader, TensorDataset
SPY = dict()


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
    

    def configure_scheduler(self, optimizer):
        """Override this method to provide a learning rate scheduler"""
        return None
    

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
        self.scheduler = None

    
    def prepare_data(self, data):
        self.train_loader = data.train_dataloader()
        self.val_loader = data.val_dataloader()


    def prepare_model(self, model):
        self.model = model.to(self.device)
        model.set_trainer(self)
        self.optimizer = model.configure_optimizers()
        self.scheduler = model.configure_scheduler(self.optimizer)


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


    def _step_scheduler(self):
        """Step the learning rate scheduler if available"""
        if self.scheduler is not None:
            self.scheduler.step()

        
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
            self._step_scheduler()
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
    def __init__(self, batch_size=64, resize=(28, 28), root='../data/fashion-mnist', num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.resize = resize
        self.root = root
        self.num_workers = num_workers
        self.mean = [0.0]  # Grayscale image, no normalization
        self.std = [1.0]
        
        # Định nghĩa transform
        trans = [transforms.ToTensor()]
        if resize is not None:
            trans.insert(0, transforms.Resize(resize))
        trans = transforms.Compose(trans)
        
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
    def __init__(self, batch_size=64, resize=(32, 32), root='../data/cifar-10', num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.root = root
        self.num_workers = num_workers
        self.mean = [0.491, 0.482, 0.447]
        self.std = [0.247, 0.243, 0.262]
        
        # Define transform with data augmentation for training
        train_list = []
        if resize is not None:
            train_list.append(transforms.Resize(resize))
            train_list.append(transforms.RandomHorizontalFlip())
            train_list.append(transforms.RandomCrop(resize[0], padding=4))
        
        train_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        train_trans = transforms.Compose(train_list)
        
        test_list = []
        if resize is not None:
            test_list.append(transforms.Resize(resize))
            
        test_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        test_trans = transforms.Compose(test_list)
        
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


class CIFAR100(DataModule):
    """CIFAR-100 dataset from torchvision"""
    def __init__(self, batch_size=64, resize=(32, 32), root='../data/cifar-100', num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.root = root
        self.num_workers = num_workers
        self.mean = [0.507, 0.487, 0.441]
        self.std = [0.267, 0.256, 0.276]
        
        # Define transform (giống CIFAR10)
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
        
        # Download + load dataset (CIFAR100 thay vì CIFAR10)
        self.train = torchvision.datasets.CIFAR100(
            root=self.root, train=True, transform=train_trans, download=True)
        self.val = torchvision.datasets.CIFAR100(
            root=self.root, train=False, transform=test_trans, download=True)
    

    def text_labels(self, indices):
        """Return text labels for indices (accepts single int or list)"""
        labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'cupboard', 'curtain', 'curve', 'cushion', 'daisy', 'dam', 'dance', 'danger', 'daring', 'deer', 'defense', 'device', 'diamond', 'diary', 'dice', 'diet', 'difference', 'digital', 'diminish', 'dinosaur', 'direct', 'dirt', 'disagree', 'discover', 'disease', 'dish', 'dismiss', 'disorder', 'display', 'distance', 'divide', 'dock', 'doctor', 'dog', 'doll', 'dolphin', 'dome', 'dominant', 'dominate', 'done', 'donkey', 'donor', 'door', 'dose', 'double', 'dove', 'down', 'dozen', 'draft', 'dragon', 'dragonfly', 'drain', 'drama', 'drastic', 'draw', 'dream', 'dress', 'drift', 'drill', 'drink', 'drip', 'drive', 'driver', 'drop', 'drove', 'drown', 'drum', 'drunk', 'dry', 'duck', 'dumb', 'dune', 'dungeon', 'duplicate', 'dusk', 'dust', 'duty', 'dwarf', 'dwell', 'eagle', 'early', 'earn', 'earth', 'easily', 'east', 'easy', 'echo', 'eclipse', 'ecology', 'economy', 'edge', 'edit', 'educate', 'effort', 'egg', 'eight', 'either', 'elbow', 'elder', 'electric', 'elegant', 'element', 'elephant', 'elevator', 'elite', 'elk', 'elm', 'else', 'email', 'embark', 'embarrass', 'embassy', 'ember', 'emblem', 'embrace', 'emerge', 'emerald', 'emergency', 'emery', 'emotion', 'emperor', 'emphasis', 'empire', 'employ', 'empty', 'enable', 'enact', 'encounter', 'encourage', 'end', 'endanger', 'endear', 'ending', 'endless', 'endorse', 'endure', 'enemy', 'energy', 'enforce', 'engage', 'engine', 'enhance', 'enjoy', 'enlarge', 'enough', 'enrage', 'enrich', 'enroll', 'ensemble', 'ensure', 'enter', 'entertain', 'enthusiasm', 'entice', 'entire', 'entry', 'envelope', 'environment', 'epic', 'episode', 'equal', 'equally', 'equate', 'equilibrium', 'equip', 'equity', 'era', 'erase', 'erect', 'ermine', 'erode', 'erosion', 'error', 'erudite', 'erupt', 'escape', 'eschew', 'escort', 'escrow', 'esophagus', 'esoteric', 'essay', 'essence', 'estate', 'esteem', 'ester', 'estimate', 'estrange', 'estuary', 'eternal', 'ether', 'ethical', 'ethnic', 'ethos', 'etiquette', 'etymology', 'eucalyptus', 'eulogize', 'eulogy', 'eunuch', 'euphemism', 'euphonic', 'euphony', 'eureka', 'european', 'evacuate', 'evade', 'evaluate', 'evangelic', 'evangelist', 'evaporate', 'evasion', 'eve', 'even', 'evening', 'evenly', 'event', 'eventful', 'eventual', 'ever', 'everglade', 'evergreen', 'everlasting', 'every', 'everyday', 'everyone', 'everything', 'everywhere', 'evict', 'evidence', 'evident', 'evil', 'evince', 'eviscerate', 'evoke', 'evolution', 'evolve', 'ewe', 'exact', 'exacting', 'exaction', 'exactly', 'exaggerate', 'exalt', 'examine', 'example', 'exasperate', 'excavate', 'exceed', 'excel', 'excellence', 'excellent', 'except', 'exception', 'excerpt', 'excess', 'exchange', 'exchequer', 'excise', 'excision', 'excitable', 'excitation', 'excite', 'exclaim', 'exclamation', 'exclude', 'exclusion', 'exclusive', 'excommunicate', 'excrement', 'excrescence', 'excretion', 'excrete', 'excruciating', 'exculpate', 'excursion', 'excuse', 'execute', 'execution', 'executive', 'executor', 'exegesis', 'exemplar', 'exemplary', 'exemplify', 'exempt', 'exemption', 'exercise', 'exert', 'exertion', 'exhale', 'exhaust', 'exhaustible', 'exhaustion', 'exhaustive', 'exhibit', 'exhibition', 'exhilarate', 'exhort', 'exhortation', 'exhume', 'exigency', 'exigent', 'exiguity', 'exiguous', 'exile', 'exist', 'existence', 'existent', 'existential', 'existentialism', 'existing', 'exit', 'exiting', 'exodus', 'exonerate', 'exorbitance', 'exorbitant', 'exorcise', 'exorcism', 'exorcist', 'exotic', 'exotica', 'expand', 'expander', 'expanse', 'expansion', 'expansive', 'expatiate', 'expatriate', 'expect', 'expectancy', 'expectant', 'expectation', 'expectorant', 'expectorate', 'expediency', 'expedient', 'expedite', 'expedition', 'expeditious', 'expel', 'expend', 'expendable', 'expenditure', 'expense', 'expensive', 'experience', 'experiential', 'experiment', 'experimental', 'experimentation', 'experimenter', 'expert', 'expertise', 'expiate', 'expiation', 'expiration', 'expiratory', 'expire', 'expiry', 'explain', 'explanation', 'explanatory', 'expletive', 'explicit', 'explicitness', 'explode', 'exploit', 'exploitation', 'exploiter', 'exploration', 'exploratory', 'explore', 'explorer', 'explosion', 'explosive', 'expo', 'exponent', 'exponential', 'export', 'exporter', 'expose', 'exposition', 'expostulation', 'exposure', 'expound', 'express', 'expression', 'expressionism', 'expressionist', 'expressionless', 'expressive', 'expressively', 'expressway', 'expropriate', 'expropriation', 'expulsion', 'expunge', 'expurgate', 'expurgation', 'exquisite', 'exquisiteness', 'extant', 'extemporaneous', 'extemporarily', 'extemporize', 'extend', 'extender', 'extensible', 'extension', 'extensional', 'extensive', 'extensiveness', 'extensor', 'extent', 'extenuate', 'extenuation', 'exterior', 'exteriorly', 'exterminate', 'extermination', 'exterminator', 'external', 'externality', 'externally', 'extinct', 'extinction', 'extinguish', 'extinguisher', 'extirpate', 'extirpation', 'extol', 'extoll', 'extort', 'extortion', 'extortionist', 'extra', 'extract', 'extraction', 'extractor', 'extracurricular', 'extraditable', 'extradite', 'extradition', 'extrajudicial', 'extramarital', 'extramural', 'extraneous', 'extraordinarily', 'extraordinariness', 'extraordinary', 'extrapolate', 'extrapolation', 'extrasensory', 'extravagance', 'extravagant', 'extravaganza', 'extravasate', 'extravasation', 'extreme', 'extremely', 'extremism', 'extremist', 'extremity', 'extricate', 'extrication', 'extroversion', 'extrovert', 'extrude', 'extrusion', 'exuberance', 'exuberant', 'exudation', 'exude', 'exult', 'exultancy', 'exultant', 'exultation', 'eye', 'eyeball', 'eyebright', 'eyebrow', 'eyecup', 'eyed', 'eyedness', 'eyedropper', 'eyeful', 'eyelash', 'eyelet', 'eyelid', 'eyeopener', 'eyepiece', 'eyeshot', 'eyesight', 'eyesore', 'eyestrain', 'eyetooth', 'eyewash', 'eyewater', 'eyewink', 'eyrie', 'eyre', 'ezra']
        if isinstance(indices, int):
            return labels[indices]
        return [labels[int(i)] for i in indices]
    

    def get_dataloader(self, train):
        """Get dataloader for training or validation"""
        data = self.train if train else self.val
        return DataLoader(data, self.batch_size, shuffle=train, num_workers=self.num_workers)


def visualize_prediction(model, data, num_examples=8, trainer=None, save_path=None):
    """Function for visualizing predictions if images for image classification tasks
    
    Args:
        model: Trained model
        data: Data module with validation set
        num_examples: Number of examples to visualize
        trainer: Trainer instance for device info
        save_path: Path to save the figure (e.g., 'predictions.png'). If None, only display
    """
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
    
    # Save if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def try_all_gpus():
    """Try all existing GPUs"""
    def gpu(i=0):
        return torch.device(f'cuda:{i}')
    devices = [gpu(i) for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


class Vocab:
    """Build vocabulary for language models and other convenient functions"""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten 2D list if necessary
        if tokens and isinstance(tokens[0], list):
            counter = collections.Counter()
            for line in tokens:
                counter.update(line)
        else:
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
        loss_fn = nn.CrossEntropyLoss(reduction='none')
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
                 emb_dropout, blk_dropout, lr=0.1, use_bias=False, num_classes=10, 
                 optimizer_type='sgd', weight_decay=0.0):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        self.optimizer_type = optimizer_type
        self.weight_decay = weight_decay
        num_steps = self.patch_embedding.num_patches + 1
        # Positional embeddings are learnable
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


    def configure_optimizers(self):
        """Choose optimizer based on setting"""
        if self.optimizer_type == 'adamw':
            return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:  # Default: SGD
            return torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


import hashlib
import requests

def download(url, folder='../data', sha1_hash=None):
    """Download a file to folder and return its local path"""
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, url.split('/')[-1])

    def get_file_sha1(path):
        sha1 = hashlib.sha1()
        with open(path, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data: break
                sha1.update(data)
        return sha1.hexdigest()
    # Check if file exists
    if os.path.exists(fname):
        actual_hash = get_file_sha1(fname)
        # Check if cache hits
        if sha1_hash and actual_hash == sha1_hash:
            return fname
        if sha1_hash is None:
            SPY[url] = actual_hash
            return fname
    # Download if cache miss or file doesn't exist
    print(f"Downloading {fname} from {url}...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(fname, 'wb') as f:        # wb = write binary
        f.write(r.content)
    # Update sha1_hash into dictionary
    downloaded_hash = get_file_sha1(fname)
    SPY[url] = downloaded_hash
     
    return fname


import zipfile
import tarfile

def download_extract(name, folder='../data', in_folder=None, sha1_hash=None):
    """Download and extract a zip/tar file"""
    fname = download(name, folder, sha1_hash)
    base_dir = os.path.dirname(fname)
    # Extract filename into 2 parts: filename without extension and extension
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
        # Filter out problematic mac-specific files
        members = [m for m in fp.infolist() if not m.filename.startswith('__MACOSX/') 
                   and 'Icon\r' not in m.filename]
        fp.extractall(base_dir, members=members)
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
        fp.extractall(base_dir)
    else:
        assert False, 'Only zip/tar files can be extracted!'
    
    return os.path.join(base_dir, in_folder) if in_folder else data_dir
    

import random

def subsample(sentences, vocab):
    """Subsample high frequency words"""
    # Exclude unknown tokens ('<unk>')
    sentences = [[token for token in line if vocab[token] != vocab.unk] for line in sentences]
    counter = collections.Counter([token for line in sentences for token in line])
    num_tokens = sum(counter.values())

    # Return True if 'token'
    def keep(token):
        return (random.uniform(0, 1) < math.sqrt(1e-4 / counter[token] * num_tokens))
    
    return ([[token for token in line if keep(token)] for line in sentences], counter)


def get_centers_and_contexts(corpus, max_window_size):
    """Return center words and context words in skip-gram model"""
    centers, contexts = [], []

    for line in corpus:
        # Each sentence must have at least 2 words to form a "sentence pair - word pair"
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size), min(len(line), i + 1 + window_size)))
            # Exclude the center word from the context word
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])

    return (centers, contexts)


class RandomGenerator:
    """Randomly draw among {1, 2,..., n} according to n sampling weights"""
    def __init__(self, sampling_weights):
        # Exclude
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0
    

    def draw(self):
        if self.i == len(self.candidates):
            # Cache 'k' random sampling results
            self.candidates = random.choices(self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]
    

def get_negatives(all_contexts, vocab, counter, K):
    """Return noise words in negative sampling"""
    # Sampling weights for words with indices 1, 2, 3,..., (index 0 is the exclude unknown token)
    # in the vocabulary
    sampling_weights = [counter[vocab.to_tokens(i)] ** 0.75 for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)

    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # Exclude context words from noise words
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)

    return all_negatives


def batchify(data):
    """Return a minibatch of examples for skip-gram with negative sampling"""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (torch.reshape(torch.tensor(centers), (-1, 1)), torch.tensor(
        contexts_negatives), torch.tensor(masks), torch.tensor(labels))


def read_ptb():
    """Load the PTB dataset into a list of text lines"""
    data_dir = download_extract('http://d2l-data.s3-accelerate.amazonaws.com/ptb.zip', '../../data')
    # Read the training set
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]


def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory"""
    sentences = read_ptb()
    vocab = Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(corpus, max_window_size)
    all_negatives = get_negatives(all_contexts, vocab, counter, num_noise_words)

    
    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives


        def __getitem__(self, index):        # Allow an object to behave like a List or a Dictionary.
            return (self.centers[index], self.contexts[index], self.negatives[index])


        def __len__(self):
            return len(self.centers)
        
    
    dataset = PTBDataset(all_centers, all_contexts, all_negatives)
    # collate_fn: a custom function that can be passed to the DataLoader to teach it how to package
    # individual samples into a batch.
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, collate_fn=batchify)

    return data_iter, vocab


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    if legend:
        axes.legend(legend)
    axes.grid()


from matplotlib_inline import backend_inline


class Animator:
    """For plotting data in animation"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear',
                 yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        backend_inline.set_matplotlib_formats('svg')
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a function to capture arguments
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, 
                                            xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts


    def add(self, x, y):
        """Add multiple data points into the figure"""
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmts in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmts)
        self.config_axes()
        display(self.fig)
        clear_output(wait=True)
        
    
class Accumulator:
    """For accumulating sums over n variables"""
    def __init__(self, n):
        self.data = [0.0] * n
    

    def add(self, *arg):
        self.data = [a + float(b.item()) if isinstance(b, torch.tensor)
                     else a + float(b) for a, b in zip(self.data, arg)]


    def reset(self):
        self.data = [0.0] * len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]
    

class TokenEmbedding:
    """Download token embeddings from pretrained GloVe or fastText model"""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}


    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = download_extract(embedding_name, folder='../../data')
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1: ]]
                # Skip the header information
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        # Vector representation for '<unk>'
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec)
    

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx) for token in tokens]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        return vecs
    

    def __len__(self):
        return len(self.idx_to_token)
    

def accuracy(Y_hat, Y, averaged=True):
        """Compute accuracy. Y_hat and Y can have different shapes"""
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        Y = Y.reshape(-1)
        pred = torch.argmax(Y_hat, dim=1)
        compare = (pred == Y).type(torch.float32) 
        return compare.mean() if averaged else compare.sum()


def evaluate_accuracy(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset - evaluation mode"""
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metrics = Accumulator(2)   # number of correct predictions, number of predictions

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metrics.add(accuracy(net(X), y, False), y.numel())
    return metrics[0] / metrics[1]


def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None, ylim=None,
         xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'),
         figsize=(3.5, 2.5), axes=None):
    """Plot data points"""
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list) and not hasattr(X[0], "__len__"))
    
    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(Y) != len(X):
        X = X * len(Y)

    backend_inline.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize

    if axes is None:
        axes = plt.gca()        # get current axes
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x, y, fmt) if len(x) else axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """"Get tokens of the BERT input sequence and their segment ID"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 and 1 are marking segment A and B, respectively
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


class BERTEncoder(nn.Module):
    """BERT Encoder"""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout, max_len=1000, **kwargs):
        super().__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        # In BERT, positional embedding are learnable
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, True
            ))

    def forward(self, tokens, segments, valid_lens):
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
    

class MaskLM(nn.Module):
    """Masked Language Model implement for BERT"""
    def __init__(self, vocab_size, num_hiddens, **kwargs):
        super().__init__(**kwargs)
        self.mlp = nn.Sequential(
            nn.LazyLinear(num_hiddens),
            nn.ReLU(),
            nn.LayerNorm(num_hiddens),
            nn.LazyLinear(vocab_size)
        )


    def forward(self, X, pred_positions):
        # pred_positions shape: (batch_size, num_pred_positions)
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `torch.tensor([0, 0, 0, 1, 1, 1])`
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        # masked_X shape: (batch_size * num_pred_positions, num_hiddens)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
    

class NextSequencePred(nn.Module):
    """The next sequence prediction task of BERT"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output = nn.LazyLinear(2)

    
    def forward(self, X):
        # X size: (batch_size, num_hiddens) -> X is <cls> token
        return self.output(X)
    

class BERTModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, max_len=1000):
        super().__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens,
                                   num_heads, num_blks, dropout, max_len)
        self.hidden = nn.Sequential(
            nn.LazyLinear(num_hiddens), 
            nn.Tanh()
            )
        self.mlm = MaskLM(vocab_size, num_hiddens)
        self.nsp = NextSequencePred()


    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat


def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wikitext-2-v1-train.txt')
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # Uppercase letters are converted to lowercase
    paragraphs = [line.strip().lower().split(' . ') 
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    # paragraphs is a list of list of lists
    return paragraphs


def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next


def _get_nsp_data_from_paragraph(paragraph, paragraphs, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs
        )
        # 2 <sep> tokens and 1 <cls> token
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph


def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab):
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    random.shuffle(candidate_pred_positions)

    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80% of the time -> replace the word with the '<mask>' token
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10% of the time -> keep the word unchanged
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10% of the time -> replace the word with a random word
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))
    
    return mlm_input_tokens, pred_positions_and_labels


def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # 'tokens' is a list of string
    for i, token in enumerate(tokens):
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 15% of random tokens are predicted in the masked language modeling task
    mlm_num_preds = max(1, round(len(tokens) * .15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, mlm_num_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]

    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


def _pad_BERT_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * .15)
    all_token_ids, all_segments, valid_lens = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    
    for (
        token_ids, 
        pred_positions, 
        mlm_pred_label_ids, 
        segments, 
        is_next
    ) in examples:
        all_token_ids.append(torch.tensor(
            token_ids + [vocab['<pad>']] * (max_len - len(token_ids)), 
            dtype=torch.long))
        all_segments.append(torch.tensor(
            segments + [0] * (max_len - len(token_ids)),
            dtype=torch.long))
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(
            pred_positions + [0] * (max_num_mlm_preds - len(mlm_pred_label_ids)),
            dtype=torch.long
        )) 
        all_mlm_weights.append(torch.tensor(
            [1.0] * len(pred_positions) + [0.0] * (max_num_mlm_preds - len(pred_positions)),
            dtype=torch.float32
        ))
        all_mlm_labels.append(torch.tensor(
            mlm_pred_label_ids + [0] * (max_num_mlm_preds - len(mlm_pred_label_ids)),
            dtype=torch.long
        ))
        nsp_labels.append(torch.tensor(
            is_next, dtype=torch.long
        ))
    
    return (all_token_ids, all_segments, valid_lens, all_pred_positions, 
            all_mlm_weights, all_mlm_labels, nsp_labels)


def tokenize(lines, token='word'):
    """Split text lines into word or character tokens"""
    assert token in ('word', 'character'), 'Unknown token type: ' + token
    return [line.split() if token == 'word' else list(line) for line in lines]


class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        paragraphs = [tokenize(paragraph) for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        self.vocab = Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<sep>', '<mask>', '<cls>'])
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, max_len
            ))
        # Get data for the masked language modeling task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab) + (segments, is_next))
                    for (tokens, segments, is_next) in examples]
        # Pad BERT inputs
        (self.all_token_ids, self.all_segments, self.valid_lens, self.all_pred_positions, 
         self.all_mlm_weights, self.all_mlm_labels, self.nsp_labels) = _pad_BERT_inputs(
             examples, max_len, self.vocab
         )
        

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx], self.valid_lens[idx], 
                self.all_pred_positions[idx], self.all_mlm_weights[idx], 
                self.all_mlm_labels[idx], self.nsp_labels[idx])
    

    def __len__(self):
        return len(self.all_token_ids)
     

def load_data_wiki(batch_size, max_len):
    """Load the WikiText-2 dataset"""
    data_dir = "../../data/wikitext-2"
    os.makedirs(data_dir, exist_ok=True)
    train_ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")

    with open(os.path.join(data_dir, "wikitext-2-v1-train.txt"), "w", encoding="utf-8") as f:
        for line in train_ds["text"]:
            f.write(line + "\n")
    
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True)
    return train_iter, train_set.vocab


def _get_batch_loss_BERT(net, loss, vocab_size, tokens_X, segments_X, 
                         valid_lens_x, pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X, 
                                  valid_lens_x, pred_positions_X)
    # Compute masked language model loss
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) \
                * mlm_weights_X.reshape(-1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-9)
    # Compute the next sequence prediction loss
    nsp_l = loss(nsp_Y_hat.reshape(-1, 2), nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l


def train_BERT(train_iter, net, loss, vocab_size, devices, num_steps):
    net(*next(iter(train_iter))[:4])
    net = nn.DataParallel(net, device_ids=devices)
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step = 0
    animator = Animator(xlabel='step', ylabel='loss', xlim=[1, num_steps],
                            legend=['mlm', 'nsp'])
    # mlm loss, nsp loss, count
    metrics = Accumulator(3)
    num_steps_reached = False

    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X, \
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])

            trainer.zero_grad()
            mlm_l, nsp_l, l = _get_batch_loss_BERT(
                net, loss, vocab_size, tokens_X, segments_X,
                valid_lens_x, pred_positions_X, mlm_weights_X, 
                mlm_Y, nsp_y
            )
            l.backward()
            trainer.step()
            metrics.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            animator.add(step + 1, (metrics[0] / metrics[2], metrics[1] / metrics[2]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f"MLM loss: {metrics[0] / metrics[2]:.3f}", 
          f"NSP loss: {metrics[1] / metrics[2]:.3f}")
    

def truncate_pad(line, num_steps, padding_token):
    """Truncate and padding tokens in a line"""
    # Truncate
    if len(line) > num_steps:
        return line[:num_steps]
    # Padding
    else:
        return line + [padding_token] * (num_steps - len(line))
    

def read_imdb(data_dir, is_train):
    """Read the IMDb review dataset"""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test', label)
        for file in tqdm(os.listdir(folder_name), desc=f'Reading {label} reviews'):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels


def load_data_imdb(batch_size, num_steps=500):
    """Return data iterator and the vocabulary of the IMDb film review dataset"""
    data_dir = download_extract('http://d2l-data.s3-accelerate.amazonaws.com/aclImdb_v1.tar.gz',
                                '../../data/aclImdb', 'aclImdb', 
                                '01ada507287d82875905620988597833ad4e0903')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = tokenize(train_data[0])
    test_tokens = tokenize(test_data[0])
    vocab = Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_features, torch.tensor(train_data[1])),
        batch_size=batch_size,
        shuffle=True
    )
    test_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_features, torch.tensor(test_data[1])),
        batch_size=batch_size,
        shuffle=False
    )
    return train_iter, test_iter, vocab


def train_batch(net, X, y, loss, trainer, devices):
    """Train a minibatch on multiple GPUs"""
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else: 
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss = l.sum()
    train_acc = accuracy(pred, y, False)
    return train_loss, train_acc


def train(net, train_iter, test_iter, loss, trainer, num_epochs, devices=try_all_gpus()):
    """Train a model on multiple GPUs"""
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    num_batches = len(train_iter)
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], 
                        legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        # train loss, train_acc, number of examples, number of predictions
        metrics = Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            train_loss, train_acc = train_batch(
                net, features, labels, loss, trainer, devices)
            metrics.add(train_loss, train_acc, labels.shape[0], labels.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, 
                             [metrics[0] / metrics[2], metrics[1] / metrics[3], None])
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, [None, None, test_acc])
            
    print(f'Train loss: {metrics[0] / metrics[2]:.3f}\n'
          f'Train acc: {metrics[1] / metrics[3]:.3f}\n'
          f'Test acc: {test_acc:.3f}')
    

def predict_sentiment(net, vocab, sentence):
    """Predict the sentiment of a text sequence"""
    sequence = torch.tensor(
        vocab[sentence.split()], device='cuda' if torch.cuda.is_available() else 'cpu')
    preds = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if preds == 1 else 'negative'


def read_snli(data_dir, is_train):
    """Read the SNLI dataset into premises, hypotheses, and labels"""
    def extract_text(s):
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(
        data_dir, 'snli_1.0_train.txt' if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]

    premises, hypotheses, labels = [], [], []
    for row in tqdm(rows, desc="Processing SNLI dataset"):
        if row[0] in label_set:
            premises.append(extract_text(row[1]))
            hypotheses.append(extract_text(row[2]))
            labels.append(label_set[row[0]])

    return premises, hypotheses, labels


class SNLIDataset(torch.utils.data.Dataset):
    """A customized dataset class for SNLI dataset"""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        premise_tokens = tokenize(dataset[0])
        hypothesis_tokens = tokenize(dataset[1])
        if vocab is None:
            self.vocab = Vocab(
                premise_tokens + hypothesis_tokens, min_freq=5, reserved_tokens=['<pad>'])
        else: 
            self.vocab = vocab
        self.premises = self._pad(premise_tokens)
        self.hypotheses = self._pad(hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])


    def _pad(self, lines):
        return torch.tensor([truncate_pad(
                self.vocab[line], 
                num_steps=self.num_steps, 
                padding_token=self.vocab['<pad>']) for line in lines])
    

    def __getitem__(self, index):
        return (self.premises[index], self.hypotheses[index]), self.labels[index]
    

    def __len__(self):
        return len(self.labels)
    

def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary"""
    data_dir = download_extract(
        'https://nlp.stanford.edu/projects/snli/snli_1.0.zip', 
        '../../data', sha1_hash='9fcde07509c7e87ec61c640c1b2753d9041758e4')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, vocab=train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_iter, test_iter, train_set.vocab


def predict_snli(net, vocab, premise, hypothesis):
    """Predict the logical relationship between the premise and the hypothesis"""
    net.eval()
    premise = torch.tensor(vocab[premise], 
                           device = 'cuda' if torch.cuda.is_available() else 'cpu')
    hypothesis = torch.tensor(vocab[hypothesis],
                              device = 'cuda' if torch.cuda.is_available() else 'cpu')
    label = torch.argmax(net([
        premise.reshape(1, -1), hypothesis.reshape(1, -1)]), dim=-1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 else 'neutral'