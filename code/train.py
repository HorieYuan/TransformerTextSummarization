import os
import pickle

import gluonnlp as nlp
import mxnet as mx
from mxboard import SummaryWriter
from mxnet import autograd, gluon, init, nd
from mxnet import util as mxutil
from mxnet.gluon import loss as gloss
from mxnet.gluon import utils as gutils

import hyper_parameters as hp
from summarize import idx_to_tokens, summarize
from Transformer import Transformer
from utils import DatasetAssiant, PairsDataLoader, PairsDataset

N_EPOCHS = hp.N_EPOCHS
BATCH_SIZE = hp.BATCH_SIZE

MAX_SOURCE_LEN = hp.MAX_SOURCE_LEN
MAX_TARGET_LEN = hp.MAX_TARGET_LEN

EMBED_SIZE = hp.EMBED_SIZE
MODEL_DIM = hp.MODEL_DIM

LR = hp.LR

HEAD_NUM = hp.HEAD_NUM
LAYER_NUM = hp.LAYER_NUM
FFN_DIM = hp.FFN_DIM

DROPOUT = hp.DROPOUT
ATT_DROPOUT = hp.ATT_DROPOUT
FFN_DROPOUT = hp.FFN_DROPOUT

TRAIN_DATA_PATH = hp.TRAIN_DATA_PATH
VOCAB_PATH = hp.VOCAB_PATH

# CTX = try_gpu()
gpu_count = mxutil.get_gpu_count()
CTX = [mx.gpu(i) for i in range(gpu_count)]


BOS = hp.BOS
EOS = hp.EOS


def batch_loss(transformer, X, Y, label, vocab, loss):
    batch_size = X.shape[0]
    Y_h = transformer(X, Y)

    l = loss(Y_h, label).sum()

    print(idx_to_tokens(nd.argmax(Y_h[0], axis=-1), vocab))

    return l / batch_size


def train(transformer, data_iter, lr, num_epochs, vocab, ctx):
    print('start training')
    print('ctx:', ctx)

    trainer = gluon.Trainer(transformer.collect_params(), 'adam', {'learning_rate': lr})
    loss = gloss.SoftmaxCrossEntropyLoss()

    best_epoch = 0
    best_loss = float('Inf')
    sw = SummaryWriter(logdir='../logs', flush_secs=5)

    for epoch in range(num_epochs):
        l_sum = 0.0
        for i, data in enumerate(data_iter):
            X, Y, label, X_valid_len, Y_valid_len = data
            # X = X.as_in_context(ctx)
            # Y = Y.as_in_context(ctx)
            # label = label.as_in_context(ctx)
            gpu_Xs = gutils.split_and_load(X, ctx, even_split=False)
            gpu_Ys = gutils.split_and_load(Y, ctx, even_split=False)
            gpu_labels = gutils.split_and_load(label, ctx, even_split=False)

            with autograd.record():
                # l = batch_loss(transformer, X, Y, vocab, loss)
                ls = [batch_loss(transformer, gpu_X, gpu_Y, gpu_label, vocab, loss)
                      for gpu_X, gpu_Y, gpu_label in zip(gpu_Xs, gpu_Ys, gpu_labels)]

            # l.backward()
            b_loss = 0.0
            for l in ls:
                l.backward()
                b_loss += l.asscalar()
            trainer.step(X.shape[0])
            nd.waitall()

            l_sum += b_loss
            if i % 100 == 0:
                info = "epoch %d, batch %d, batch_loss %.3f" % (epoch, i, b_loss)
                print(info)
                sw.add_scalar(tag='batch_loss', value=b_loss, global_step=i)

        cur_loss = l_sum / len(data_iter)
        # 保存模型
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_epoch = epoch
            if not os.path.exists('../model'):
                os.mkdir('../model')
            transformer.save_parameters('../model/transformer' + str(epoch) + '.params')

        info = "epoch %d, loss %.3f, best_loss %.3f, best_epoch %d" % (
            epoch, cur_loss, best_loss, best_epoch)
        print(info)
        sw.add_scalar(tag='loss', value=cur_loss, global_step=epoch)


def main(data_path):
    dataset = PairsDataset(data_path)
    _, vocab = nlp.model.get_model('bert_12_768_12', dataset_name='wiki_cn_cased',
                                   ctx=CTX, pretrained=True, use_pooler=False, use_decoder=False, use_classifier=False)
    assiant = DatasetAssiant(vocab, vocab, MAX_SOURCE_LEN, MAX_TARGET_LEN)
    dataloader = PairsDataLoader(dataset, BATCH_SIZE, assiant)

    # with open(VOCAB_PATH, 'wb') as fw:
    #     pickle.dump(vocab, fw)
    NWORDS = len(vocab)
    print(NWORDS)
    transformer = Transformer(vocab, vocab, EMBED_SIZE, MODEL_DIM,
                              HEAD_NUM, LAYER_NUM, FFN_DIM, DROPOUT, ATT_DROPOUT, FFN_DROPOUT, CTX)
    transformer.initialize(init.Xavier(), ctx=CTX)
    train(transformer, dataloader, LR, N_EPOCHS, vocab, CTX)


if __name__ == "__main__":
    main(TRAIN_DATA_PATH)
