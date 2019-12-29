import gluonnlp as nlp
import mxnet as mx
from mxnet import gluon, nd, test_utils

import hyper_parameters as hp

# 文本长
MAX_SOURCE_LEN = hp.MAX_SOURCE_LEN
MAX_TARGET_LEN = hp.MAX_TARGET_LEN
VOCAB_SIZE = hp.VOCAB_SIZE
PAD = hp.PAD
BOS = hp.BOS
EOS = hp.EOS


class PairsDataset(gluon.data.Dataset):
    def __init__(self, data_path, **kwargs):
        super(PairsDataset, self).__init__(**kwargs)
        self.tokenizer = nlp.data.JiebaTokenizer()
        self.sources, self.targets = self._load_data(data_path)
        self.vocab

    def _load_data(self, data_path):
        data_source = []
        data_summa = []
        tokens = []
        data_count = 0
        error_data_count = 0

        with open(data_path, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.rstrip('\n')
                line = line.split('\t')
                if len(line) == 2:
                    source = self.tokenizer(line[0])
                    target = self.tokenizer(line[1])
                    tokens += list(set(source))
                    tokens += list(set(target))
                    data_source.append(source)
                    data_summa.append(target)
                    data_count += 1
                else:
                    error_data_count += 1
        counter = nlp.data.count_tokens(tokens)
        self.vocab = nlp.Vocab(counter, max_size=VOCAB_SIZE,
                               padding_token=PAD, bos_token=BOS, eos_token=EOS)
        print(data_count, '条数据, ', error_data_count, '条读取错误')
        return data_source, data_summa

    def __getitem__(self, item):
        return self.sources[item], self.targets[item]

    def __len__(self):
        return len(self.sources)

    def get_vocab(self):
        return self.vocab


class DatasetAssiant():
    def __init__(self, src_vocab=None, tgt_vocab=None, max_src_len=None, max_tgt_len=None):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.bos_token = BOS
        self.eos_token = EOS

    def pair_sentence_process(self, *src_and_tgt):
        src, tgt = src_and_tgt

        if self.max_src_len and len(src) > self.max_src_len - 2:
            src = src[0:self.max_src_len]
        if self.max_tgt_len and len(tgt) > self.max_tgt_len - 1:
            tgt = tgt[0:self.max_tgt_len]

        # src = [self.bos_token] + src + [self.eos_token]
        # tgt = [self.bos_token] + tgt
        # label = tgt[1:] + [self.eos_token]
        label = tgt[1:] + [self.eos_token]

        src_valid_len = len(src)
        tgt_valid_len = len(tgt)

        src = self.src_vocab[src]
        tgt = self.tgt_vocab[tgt]
        label = self.tgt_vocab[label]

        return src, tgt, label, src_valid_len, tgt_valid_len


class PairsDataLoader(object):
    def __init__(self, dataset, batch_size, assiant, shuffle=False, num_workers=3, lazy=True):
        trans_func = assiant.pair_sentence_process
        self.dataset = dataset.transform(trans_func, lazy=lazy)
        self.batch_size = batch_size
        self.src_pad_val = assiant.src_vocab[assiant.src_vocab.padding_token]
        self.tgt_pad_val = assiant.tgt_vocab[assiant.tgt_vocab.padding_token]
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.dataloader = self._build_dataloader()

    def _build_dataloader(self):
        batchify_fn = nlp.data.batchify.Tuple(
            nlp.data.batchify.Pad(pad_val=self.src_pad_val),
            nlp.data.batchify.Pad(pad_val=self.tgt_pad_val),
            nlp.data.batchify.Pad(pad_val=self.tgt_pad_val),
            nlp.data.batchify.Stack(dtype="float32"),
            nlp.data.batchify.Stack(dtype="float32"),)
        dataloader = gluon.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size,
                                           shuffle=self.shuffle, batchify_fn=batchify_fn,
                                           num_workers=self.num_workers)
        return dataloader

    def __iter__(self):
        for d in self.dataloader:
            yield d

    def __len__(self):
        return len(self.dataloader)


def convert_text_to_idx(data, word_vocab):
    def convert(x):
        return word_vocab[x]
    data = list(map(convert, data))
    data = nd.array(data)
    return data


def try_gpu():
    if len(test_utils.list_gpus()) > 0:
        return mx.gpu()
    else:
        return mx.cpu()


def main(argv):
    data_path = '/home/horie/workspace/TransformerSummarization/data/test_lcsts.tsv'
    dataset = PairsDataset(data_path)
    vocab = dataset.get_vocab()
    dataloader = PairsDataLoader(dataset, 3, DatasetAssiant(
        vocab, vocab, MAX_SOURCE_LEN, MAX_TARGET_LEN))
    for data in dataloader:
        X, Y, label, X_valid_len, Y_valid_len = data
        print(X[0])
        from summarize import idx_to_tokens
        print(idx_to_tokens(X[0], vocab))
        print(Y[0])
        print(idx_to_tokens(Y[0], vocab))
        print(label[0])
        print(idx_to_tokens(label[0], vocab))
        break


if __name__ == "__main__":
    main(0)
