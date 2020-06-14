import tensorflow.compat.v1 as tf

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
        self.y_one_hot = tf.one_hot(self.y, self.num_classes, dtype=tf.float32)

        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')

        weights = {
            'layer_0': tf.Variable(tf.random_normal([
                self.seq_max_len * self.vocab_size, 
                self.state_size
            ])),
            'layer_1': tf.Variable(tf.random_normal([
                self.state_size, 
                self.state_size
            ])),
            'layer_2': tf.Variable(tf.random_normal([
                self.state_size,
                self.num_classes
            ]))
        }

        # y = mx + b
        biases = {
            'layer_0': tf.Variable(tf.random_normal([self.state_size])), 
            'layer_1': tf.Variable(tf.random_normal([self.state_size])),
            'layer_2': tf.Variable(tf.random_normal([self.num_classes]))
        }

        x_input = tf.reshape(x_one_hot, [-1, self.seq_max_len * self.vocab_size])
        hidden = tf.matmul(x_input, weights['layer_0']) + biases['layer_0']
        hidden = tf.nn.sigmoid(hidden)  # no linealidad
        
        hidden = tf.matmul(hidden, weights['layer_1']) + biases['layer_1']
        hidden = tf.nn.sigmoid(hidden)

        self.logits = tf.matmul(hidden, weights['layer_2']) + biases['layer_2']
        self.probs = tf.nn.softmax(hidden)

        self.correct_preds = tf.equal(
            tf.argmax(self.probs, axis=1), 
            tf.argmax(self.y_one_hot, axis=1)
        )

        self.precision = tf.reduce_mean(tf.cast(self.correct_preds, tf.float32))

    def step_training(self, learning_rate=0.01):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, 
            labels=self.y_one_hot
        ))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        
        return loss, optimizer
