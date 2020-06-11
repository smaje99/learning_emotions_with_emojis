import tensorflow as tf

class LogisticClassifier:

    def __init__(self, seq_max_len, state_size, vocab_size, num_classes):
        self.seq_max_len = seq_max_len
        self.state_size = state_size
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        
    def build_model(self):
        self.x = tf.placeholder(shape=[None, self.seq_max_len], dtype=tf.int32)
        x_one_hot = tf.one_hot(self.x, self.vocab_size)
        x_one_hot = tf.cast(x_one_hot, tf.float32)

        self.y = tf.placeholder(shape=[None], dtype=tf.int32)
        self.y_one_hot = tf.one_hot(self.y, self.num_classes, dtype=tf.float32)
        
