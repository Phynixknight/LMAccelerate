import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Attention(tf.keras.layers.Layer):
    def __init__(self, word_size=512, embed_dim=64, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.query = layers.Dense(units=embed_dim)
        self.key = layers.Dense(units=embed_dim)
        self.value = layers.Dense(units=embed_dim)

    def self_attention(self, Q, K, V):
        d_k = tf.cast(self.embed_dim, dtype=tf.float32)
        score = tf.matmul(Q, K, transpose_b = True) / tf.math.sqrt(d_k)
        score = tf.nn.softmax(score, axis=-1)
        Z = tf.matmul(score, V)
        return Z

    def call(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        Z = self.self_attention(Q, K, V)
        return Z

class MultiheadAttention(tf.keras.layers.Layer):
    def __init__(self, word_size=512, embed_dim=64, n_head=8, **kwargs):
        super(MultiheadAttention, self).__init__(**kwargs)
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.proj = layers.Dense(units=embed_dim)
        self.multihead = [Attention(word_size, embed_dim) for _ in range(n_head)]

    def call(self, x):
        Z_s = tf.concat([head(x) for head in self.multihead], axis=-1)
        Z = self.proj(Z_s)
        return Z

class MultiQueryAttention(Attention):
    def __init__(self, word_size=512, embed_dim=64, n_query=8, **kwargs):
        super(MultiQueryAttention, self).__init__(word_size, embed_dim, **kwargs)
        self.n_query = n_query
        self.proj = layers.Dense(units=embed_dim)
        self.queries = [layers.Dense(units=embed_dim) for _ in range(n_query)]

    def call(self, x):
        K = self.key(x)
        V = self.value(x)
        Z_s = tf.concat([self.self_attention(query(x), K, V) for query in self.queries], axis=-1)
        Z = self.proj(Z_s)
        return Z

class GroupedQueryAttention(Attention):
    def __init__(self, word_size=512, embed_dim=64, n_grouped=4, n_query_each_group=2, **kwargs):
        super(GroupedQueryAttention, self).__init__(word_size, embed_dim, **kwargs)
        self.grouped = [MultiQueryAttention(word_size, embed_dim, n_query=n_query_each_group) for _ in range(n_grouped)]
        self.proj = layers.Dense(units=embed_dim)

    def call(self, x):
        Z_s = tf.concat([head(x) for head in self.grouped], axis=-1)
        Z = self.proj(Z_s)
        return Z
