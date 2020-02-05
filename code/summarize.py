import gluonnlp as nlp
import mxnet as mx
from gluonnlp.data import BERTTokenizer
from mxnet import nd

import hyper_parameters as hp
from Transformer import Transformer
from utils import try_gpu


BOS = hp.BOS
MAX_TARGET_LEN = hp.MAX_TARGET_LEN

EMBED_SIZE = hp.EMBED_SIZE
MODEL_DIM = hp.MODEL_DIM

HEAD_NUM = hp.HEAD_NUM
LAYER_NUM = hp.LAYER_NUM
FFN_DIM = hp.FFN_DIM

DROPOUT = hp.DROPOUT
ATT_DROPOUT = hp.ATT_DROPOUT
FFN_DROPOUT = hp.FFN_DROPOUT

CTX = mx.cpu()


def _summarize(transformer, src, tgt_vocab):
    """
    Args:
        transformer (nn.Block): model
        src (ndarray): shape:(batch_size, seq_len)
    Returns:
        ndarray: shape:(batch_size, seq_len)
    """
    tgt = nd.array([tgt_vocab[[tgt_vocab.cls_token]]] * src.shape[0])
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
    y_h = y_h.asnumpy().tolist()
    y_h = list(map(int, y_h))
    predict = vocab.to_tokens(y_h)
    return predict


def summarize(sentences, transformer, src_vocab, tgt_vocab):
    tokenzier = BERTTokenizer(src_vocab)
    sentences = tokenzier(sentences)
    sent_idx = src_vocab.to_indices(sentences)
    sent_idx = nd.array([sent_idx])
    Y_h = _summarize(transformer, sent_idx, tgt_vocab)
    Y_h = Y_h[0].asnumpy().tolist()
    Y_h = list(map(int, Y_h))
    predict = tgt_vocab.to_tokens(Y_h)
    return predict


if __name__ == "__main__":
    _, vocab = nlp.model.get_model('bert_12_768_12', dataset_name='wiki_cn_cased',
                                   ctx=CTX, pretrained=True, use_pooler=False, use_decoder=False, use_classifier=False)
    NWORDS = len(vocab)
    transformer = Transformer(vocab, vocab, EMBED_SIZE, MODEL_DIM, HEAD_NUM, LAYER_NUM,
                              FFN_DIM, DROPOUT, ATT_DROPOUT, FFN_DROPOUT, CTX)
    sentences = input("请输入文章")
    # sentences = open('test.txt').read()

    i = 10
    transformer.load_parameters(
        '/home/horie/workspace/TransformerSummarization/model/transformer' + str(i) + '.params', ctx=CTX)

    # print(sentences)
    result = summarize(sentences, transformer, vocab, vocab)
    print(i, ''.join(result).split('[SEP]')[0])
