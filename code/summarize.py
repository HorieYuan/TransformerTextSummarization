import jieba
from mxnet import nd

import hyper_parameters as hp
from utils import try_gpu


BOS = hp.BOS
MAX_TARGET_LEN = hp.MAX_TARGET_LEN
CTX = try_gpu()


def _summarize(transformer, src, tgt_vocab):
    """
    Args:
        transformer (nn.Block): model
        src (ndarray): shape:(batch_size, seq_len)
    Returns:
        ndarray: shape:(batch_size, seq_len)
    """
    tgt = nd.array([tgt_vocab[[BOS]]] * src.shape[0])
    src = src.as_in_context(CTX)
    tgt = tgt.as_in_context(CTX)
    for i in range(MAX_TARGET_LEN):
        out = transformer(src, tgt)
        out = out[:, -1, :]
        out = nd.argmax(out, axis=-1)
        out = nd.expand_dims(out, axis=-1)
        tgt = nd.concat(tgt, out, dim=-1)
    return tgt


def idx_to_tokens(y_h, vocab):
    # print(y_h.shape)
    y_h = y_h.asnumpy().tolist()
    y_h = list(map(int, y_h))
    predict = vocab.to_tokens(y_h)
    return predict


def summarize(sentences, transformer, src_vocab, tgt_vocab):
    sentences = list(jieba.cut(sentences))
    sent_idx = src_vocab.to_indices(sentences)
    sent_idx = nd.array([sent_idx])
    Y_h = _summarize(transformer, sent_idx, tgt_vocab)
    Y_h = Y_h[0].asnumpy().tolist()
    Y_h = list(map(int, Y_h))
    predict = tgt_vocab.to_tokens(Y_h)
    return predict
