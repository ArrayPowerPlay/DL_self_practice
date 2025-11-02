import collections


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
        if not isinstance(index, (list, tuple)):
            return self.idx_to_token[index]
        return [self.idx_to_token[idx] for idx in index]
        

    @property
    def unk(self):
        return self.token_to_idx['<unk>']