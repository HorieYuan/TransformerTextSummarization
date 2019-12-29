import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet.gluon import nn


class Transformer(nn.Block):
    def __init__(self, src_vocab, tgt_vocab, embedding_dim, model_dim, head_num, layer_num, ffn_dim, dropout, att_dropout, ffn_dropout, ctx=mx.cpu(), **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self._ctx = ctx
        self._embedding_dim = embedding_dim
        self._model_dim = model_dim
        self.src_pad_idx = src_vocab(src_vocab.padding_token)
        self.tgt_pad_idx = tgt_vocab(tgt_vocab.padding_token)
        self.embedding = nn.Embedding(len(tgt_vocab.idx_to_token), embedding_dim)
        with self.name_scope():
            self.encoder = Encoder(model_dim, head_num, layer_num, ffn_dim, dropout, att_dropout, ffn_dropout)
            self.decoder = Decoder(model_dim, head_num, layer_num, ffn_dim, dropout, att_dropout, ffn_dropout)
            self.linear = nn.Dense(len(tgt_vocab.idx_to_token), flatten=False, params=self.embedding.collect_params())

    def forward(self, src_idx, tgt_idx):
        # compute encoder mask
        key_mask = self._get_key_mask(src_idx, src_idx, pad_idx=self.src_pad_idx)
        src_non_pad_mask = self._get_non_pad_mask(src_idx, pad_idx=self.src_pad_idx)

        # compute decoder mask
        self_tril_mask = self._get_self_tril_mask(tgt_idx)
        self_key_mask = self._get_key_mask(tgt_idx, tgt_idx, pad_idx=self.tgt_pad_idx)
        self_att_mask = nd.greater((self_key_mask + self_tril_mask), 1)

        context_att_mask = self._get_key_mask(src_idx, tgt_idx, pad_idx=self.src_pad_idx)
        tgt_non_pad_mask = self._get_non_pad_mask(tgt_idx, pad_idx=self.tgt_pad_idx)

        # Encoder
        position = nd.array(self._position_encoding_init(src_idx.shape[1], self._model_dim), ctx=src_idx.context)
        position = nd.expand_dims(position, axis=0)
        position = nd.broadcast_axes(position, axis=0, size=tgt_idx.shape[0])
        position = position * src_non_pad_mask
        src_emb = self.embedding(src_idx)
        enc_output = self.encoder(src_emb, position, key_mask, src_non_pad_mask)

        # Decoder
        position = nd.array(self._position_encoding_init(tgt_idx.shape[1], self._model_dim), ctx=src_idx.context)
        position = nd.expand_dims(position, axis=0)
        position = nd.broadcast_axes(position, axis=0, size=tgt_idx.shape[0])
        position = position * tgt_non_pad_mask
        tgt_emb = self.embedding(tgt_idx)

        outputs = self.decoder(enc_output, tgt_emb, position, self_att_mask, context_att_mask, tgt_non_pad_mask)
        outputs = self.linear(outputs)
        return outputs

    def _get_non_pad_mask(self, seq, pad_idx=None):
        if pad_idx:
            non_pad_mask = nd.not_equal(seq, pad_idx)
        else:
            non_pad_mask = nd.not_equal(seq, 0)
        non_pad_mask = nd.expand_dims(non_pad_mask, axis=2)
        return non_pad_mask

    def _get_key_mask(self, enc_idx, dec_idx, pad_idx=None):
        seq_len = dec_idx.shape[1]
        if pad_idx:
            pad_mask = nd.not_equal(enc_idx, pad_idx)
        else:
            pad_mask = nd.not_equal(enc_idx, 0)
        pad_mask = nd.expand_dims(pad_mask, axis=1)
        pad_mask = nd.broadcast_axes(pad_mask, axis=1, size=seq_len)
        return pad_mask

    def _get_self_tril_mask(self, dec_idx):
        batch_size, seq_len = dec_idx.shape
        mask_matrix = np.ones(shape=(seq_len, seq_len))
        mask = np.tril(mask_matrix, k=0)
        mask = nd.expand_dims(nd.array(mask, ctx=dec_idx.context), axis=0)
        mask = nd.broadcast_axes(mask, axis=0, size=batch_size)
        return mask

    def _position_encoding_init(self, max_length, dim):
        """Init the sinusoid position encoding table """
        position_enc = np.arange(max_length).reshape((-1, 1))  \
            / (np.power(10000, (2. / dim) * np.arange(dim).reshape((1, -1))))
        # Apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # position_enc: (max_length, dim)
        return position_enc


class Encoder(nn.HybridBlock):
    def __init__(self, model_dim, head_num, layer_num, ffn_dim, dropout, att_dropout, ffn_dropout, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        with self.name_scope():
            self.model_dense = nn.Dense(model_dim, flatten=False, use_bias=False)
            self.encoder_layers = []
            for i in range(layer_num):
                sub_layer = EncoderLayer(model_dim, head_num, ffn_dim, dropout, att_dropout, ffn_dropout)
                self.register_child(sub_layer)
                self.encoder_layers.append(sub_layer)

    def hybrid_forward(self, F, src_emb, position, key_mask, non_pad_mask):
        # src_emb: (batch_size, seq_len, emb_dim)
        # position: (seq_len, model_dim)
        # key_mask: (batch_size, seq_len, seq_len)
        # non_pad_mask: (batch_size, seq_len, 1)

        # (batch_size, seq_len, model_dim)
        src_emb = self.model_dense(src_emb)

        # (batch_size, seq_len, model_dim)
        enc_output = src_emb + position

        for sub_layer in self.encoder_layers:
            enc_output = sub_layer(enc_output, key_mask, non_pad_mask)

        # (batch_size, seq_len, model_dim)
        return enc_output


class EncoderLayer(nn.HybridBlock):
    def __init__(self, model_dim, head_num, ffn_dim, dropout, att_dropout, ffn_dropout, ** kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        with self.name_scope():
            self.masked_attention = MultiHeadAttention(model_dim, head_num, dropout, att_dropout)
            self.feed_forward = FeedForward(model_dim, ffn_dim, ffn_dropout)

    def hybrid_forward(self, F, enc_emb, key_mask, non_pad_mask):
        # enc_emb: (batch_size, seq_len, model_dim)
        # key_mask: (batch_size, seq_len, seq_len)
        # non_pad_mask: (batch_size, seq_len, 1)

        # (batch_size, seq_len, model_dim)
        enc_output = self.masked_attention(enc_emb, enc_emb, enc_emb, key_mask)
        # (batch_size, seq_len, model_dim)
        enc_output = F.broadcast_mul(enc_output, non_pad_mask)
        # (batch_size, seq_len, model_dim)
        enc_output = self.feed_forward(enc_output)
        # (batch_size, seq_len, model_dim)
        enc_output = F.broadcast_mul(enc_output, non_pad_mask)
        return enc_output


class Decoder(nn.HybridBlock):
    def __init__(self, model_dim, head_num, layer_num, ffn_dim, dropout, att_dropout, ffn_dropout, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self._model_dim = model_dim
        with self.name_scope():
            self.enc_model_dense = nn.Dense(model_dim, flatten=False, use_bias=False)
            self.dec_model_dense = nn.Dense(model_dim, flatten=False, use_bias=False)
            self.decoder_layers = []
            for i in range(layer_num):
                sub_layer = DecoderLayer(model_dim, head_num, ffn_dim, dropout, att_dropout, ffn_dropout)
                self.register_child(sub_layer)
                self.decoder_layers.append(sub_layer)

    def hybrid_forward(self, F, enc_output, lm_output, position, self_att_mask, context_att_mask, non_pad_mask):
        # enc_output, lm_output: (batch_size, seq_len, emb_dim)
        # position: (seq_len, model_dim)
        # self_att_mask, context_att_mask: (batch_size, seq_len, seq_len)
        # non_pad_mask: (batch_size, seq_len, 1)

        # (batch_size, seq_len, model_dim)
        enc_output = self.enc_model_dense(enc_output)
        lm_output = self.dec_model_dense(lm_output)

        # (batch_size, seq_len, model_dim)
        dec_output = lm_output + position

        for sub_layer in self.decoder_layers:
            dec_output = sub_layer(enc_output, dec_output, self_att_mask, context_att_mask, non_pad_mask)

        # (batch_size, seq_len, model_dim)
        return dec_output


class DecoderLayer(nn.HybridBlock):
    def __init__(self, model_dim, head_num, ffn_dim, dropout, att_dropout, ffn_dropout, ** kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        with self.name_scope():
            self.self_masked_attention = MultiHeadAttention(model_dim, head_num, dropout, att_dropout)
            self.context_attention = MultiHeadAttention(model_dim, head_num, dropout, att_dropout)
            self.feed_forward = FeedForward(model_dim, ffn_dim, ffn_dropout)

    def hybrid_forward(self, F, enc_output, dec_emb, self_att_mask, context_att_mask, non_pad_mask):
        # enc_output, dec_emb: (batch_size, seq_len, emb_dim)
        # self_att_mask, context_att_mask: (batch_size, seq_len, seq_len)
        # non_pad_mask: (batch_size, seq_len, 1)

        # (batch_size, seq_len, model_dim)
        dec_output = self.self_masked_attention(dec_emb, dec_emb, dec_emb, self_att_mask)
        dec_output = F.broadcast_mul(dec_output, non_pad_mask)
        dec_output = self.context_attention(dec_output, enc_output, enc_output, context_att_mask)
        dec_output = F.broadcast_mul(dec_output, non_pad_mask)
        dec_output = self.feed_forward(dec_output)
        dec_output = F.broadcast_mul(dec_output, non_pad_mask)

        # (batch_size, seq_len, model_dim)
        return dec_output


class MultiHeadAttention(nn.HybridBlock):
    def __init__(self, model_dim, head_num, dropout, att_dropout, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self._model_dim = model_dim
        self._head_num = head_num
        if self._model_dim % self._head_num != 0:
            raise ValueError('In MultiHeadAttetion, the model_dim should be divided exactly'
                             ' by the number of head_num. Received model_dim={}, head_num={}'
                             .format(model_dim, head_num))
        with self.name_scope():
            self.queries_dense = nn.Dense(model_dim, use_bias=False, flatten=False, prefix="query_")
            self.keys_dense = nn.Dense(model_dim, use_bias=False, flatten=False, prefix="keys_")
            self.values_dense = nn.Dense(model_dim, use_bias=False, flatten=False, prefix="values_")
            self.att_dropout = nn.Dropout(att_dropout)
            self.dropout = nn.Dropout(dropout)
            self.LayerNorm = nn.LayerNorm()

    def hybrid_forward(self, F, queries, keys, values, mask=None):
        # queries, keys, values: (batch_size, seq_len, model_dim)
        # mask: (batch_size, seq_len, seq_len)

        # Q, K, V: (batch_size, seq_len, model_dim)
        Q = self.queries_dense(queries)
        K = self.keys_dense(keys)
        V = self.values_dense(values)

        c_dim = int(self._model_dim / self._head_num)

        # Q, K, V: (batch_size * head_num, seq_len, model_dim / head_num)
        Q = F.reshape(F.transpose(F.reshape(Q, shape=(0, 0, self._head_num, -1)),
                                  axes=(0, 2, 1, 3)), shape=(-1, 0, 0), reverse=True)
        K = F.reshape(F.transpose(F.reshape(K, shape=(0, 0, self._head_num, -1)),
                                  axes=(0, 2, 1, 3)), shape=(-1, 0, 0), reverse=True)
        V = F.reshape(F.transpose(F.reshape(V, shape=(0, 0, self._head_num, -1)),
                                  axes=(0, 2, 1, 3)), shape=(-1, 0, 0), reverse=True)

        scale = c_dim ** -0.5
        # att_score: (batch_size * head_num, seq_len, seq_len)
        att_scores = F.batch_dot(Q, K, transpose_b=True)

        # scale
        att_scores = att_scores * scale

        # mask
        if mask is not None:
            # mask: (batch_size * head_num, seq_len, seq_len)
            mask = F.reshape(F.broadcast_axes(F.expand_dims(mask, axis=1), axis=1,
                                              size=self._head_num), shape=(-1, 0, 0), reverse=True)
            padding = F.ones_like(mask) * -np.inf
            att_scores = F.where(mask, att_scores, padding)

        # att_weights: (batch_size * head_num, seq_len, seq_len)
        att_weights = F.softmax(att_scores, axis=-1)

        # outputs: (batch_size * head_num, seq_len, model_dim / head_num)
        outputs = F.batch_dot(att_weights, V)
        # outputs: (batch_size, seq_len, model_dim)
        outputs = F.reshape(F.transpose(F.reshape(outputs, shape=(-1, self._head_num, 0, 0),
                                                  reverse=True), axes=(0, 2, 1, 3)), shape=(0, 0, -1))
        outputs = self.dropout(outputs)

        # residual
        outputs = outputs + queries
        outputs = self.LayerNorm(outputs)
        # outputs: (batch_size, seq_len, model_dim)
        return outputs


class FeedForward(nn.HybridBlock):
    def __init__(self, model_dim, ffn_dim, ffn_dropout, use_bias=True, activation="relu", **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        with self.name_scope():
            self.ffn_dense = nn.Dense(ffn_dim, activation=activation, use_bias=use_bias, flatten=False)
            self.model_dense = nn.Dense(model_dim, use_bias=use_bias, bias_initializer="zeros", flatten=False)
            self.dropout = nn.Dropout(ffn_dropout)
            self.layer_norm = nn.LayerNorm()

    def hybrid_forward(self, F, x, *args):
        # x: (batch_size, seq_len, emb_dim)

        # (batch_size, seq_len, ffn_dim)
        output = self.ffn_dense(x)
        # (batch_size, seq_len, model_dim)
        output = self.model_dense(output)
        output = self.dropout(output)
        output = self.layer_norm(x + output)

        return output


# # %%
# elayer = EncoderLayer(300, 2, 600, 0.1, 0.1, 0.1)
# elayer.initialize()
# enc_emb = nd.random_uniform(shape=(64, 20, 1))
# enc_idx = nd.random_normal(shape=(64, 20))
# key_mask = Transformer._get_key_mask(None, enc_idx, enc_idx)
# non_pad_mask = Transformer._get_non_pad_mask(None, enc_idx)
# print(enc_emb.shape, key_mask.shape, non_pad_mask.shape)
# output = elayer(enc_emb, key_mask, non_pad_mask)
# print(output.shape)

# # %%
# encoder = Encoder(400, 2, 3, 600, 0.1, 0.1, 0.1)
# encoder.initialize()

# src_emb = nd.random.randn(64, 20, 500)
# position = nd.random.randn(20, 400)
# key_mask = nd.random.randn(64, 20, 20)
# non_pad_mask = nd.random.randn(64, 20, 1)
# output = encoder(src_emb, position, key_mask, non_pad_mask)
# output.shape

# # %%
# dlayer = DecoderLayer(400, 2, 600, 0.1, 0.1, 0.1)
# dlayer.initialize()

# enc_output = nd.random.randn(64, 20, 300)
# dec_emb = nd.random.randn(64, 20, 300)
# self_att_mask = nd.random.randn(64, 20, 20)
# context_att_mask = nd.random.randn(64, 20, 20)
# non_pad_mask = nd.random.randn(64, 20, 1)
# output = dlayer(enc_output, dec_emb, self_att_mask, context_att_mask, non_pad_mask)
# output.shape

# # %%
# decoder = Decoder(300, 2, 2, 600, 0.1, 0.1, 0.1)
# decoder.initialize()
# enc_output = nd.random.randn(64, 20, 300)
# lm_output = nd.random.randn(64, 20, 300)
# position = nd.random.randn(20, 300)
# self_att_mask = nd.random.randn(64, 20, 20)
# context_att_mask = nd.random.randn(64, 20, 20)
# non_pad_mask = nd.random.randn(64, 20, 1)

# output = decoder(enc_output, lm_output, position, self_att_mask, context_att_mask, non_pad_mask)
# output.shape

# # %%
