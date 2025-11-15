import os
import matplotlib.pyplot as plt
from utils.build_vocab import Vocab
import torch
from torch.utils.data import DataLoader, TensorDataset


class MachineTranslation:
    """This class preprocesses the dataset retrieved from 'https://www.manythings.org/anki/'
    to formats that can be fed into DL models supporting machine translation tasks"""
    # Format of the raw data: English + TAB + The Other Language + TAB + Attribution
    def __init__(self, path, batch_size, num_steps=9, num_train=512, num_val=128):
        super(MachineTranslation, self).__init__()
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
    

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, self.batch_size, shuffle=train)
    

    def train_dataloader(self):
        return self.get_dataloader(train=True)
    

    def val_dataloader(self):
        return self.get_dataloader(train=False)


    

        
