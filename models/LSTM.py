import tensorflow.compat.v1 as tf

class LSTM:
    
    def __init__(self, seq_max_len, state_size, vocab_size, num_classes):
        self.seq_max_len = seq_max_len
        self.state_size = state_size  # tamaño del la memeria
        self.vocab_size = vocab_size
        self.num_classes = num_classes

    def build_model(self):
        self.x= tf.placeholder(shape=[None, self.seq_max_len], dtype=tf.int32)
        x_one_hot= tf.one_hot(self.x, self.vocab_size)

        # [batch_size, seq_max_len, vocab_size]
        # seq_max_len * [batch_size, vocab_size]
        x_one_hot = tf.cast(x_one_hot, tf.float32)
        rnn_input = tf.unstack(x_one_hot, axis=1)

        self.y = tf.placeholder(shape=[None], dtype=tf.int32)
        self.y_one_hot = tf.one_hot(self.y, self.num_classes, dtype=tf.float32)
        self.batch_size = tf.placeholder(shape=[], dtype=tf.int32, name='batch_size')

        weights = {
            'layer_0': tf.Variable(tf.random_normal([self.seq_max_len, 256])),
            'layer_1': tf.Variable(tf.random_normal([256, self.num_classes]))
        }
        bias = {
            'layer_0': tf.Variable(tf.random_normal([256])), 
            'layer_1': tf.Variable(tf.random_normal([self.num_classes]))
        }

        init_state = tf.zeros([self.batch_size, self.state_size])
        cell = tf.contrib.rnn.BasicLSTMCell(self.state_size)
        self.outputs, self.final_state = tf.contrib.rnn.static_rnn(
            cell, 
            rnn_input, 
            dtype=float32
        )

        output = self.outputs[-1]
        hidden = tf.matmul(output, weights['layer_0']) + biases['layer_0']
        hidden = tf.nn.tanh(hidden)
        self.logits = tf.matmul(output, weights['layer_1']) + biases['layer_1']
        self.probs = tf.nn.softmax(hidden)  # convetir varlores en probabilidades

        self.correct_preds = tf.equal(
            tf.argmax(self.probs, axis=1), 
            tf.argmax(self.y_one_hot, axis=1)
        )
        self.precision = tf.reduce_mean(tf.cast(self.correct_preds, tf.float32))

    def step_training(self, learning_rate=0.01):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, 
            labels=self.y_one_hot
        ))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        
        return loss, optimizer
       