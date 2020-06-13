import tensorflow as tf

class MLP:

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

        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')

    def step_training(self, learning_rate=0.01):
        pass
