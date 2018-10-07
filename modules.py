import tensorflow as tf 
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder:
    """Bidirectional GRU RNN Encoder"""

    def __init__(self,hidden_size,keep_prob):
        """
            hidden_size: number of hidden size of RNN
            keep_prob: keep prob. for dropout
        """

        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.fw_rnn = rnn_cell.GRUCell(self.hidden_size)
        self.fw_rnn = DropoutWrapper(self.fw_rnn, input_keep_prob=self.keep_prob)
        self.bw_rnn = rnn_cell.GRUCell(self.hidden_size)
        self.bw_rnn = DropoutWrapper(self.bw_rnn, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks, scopename):
        with vs.variable_scope(scopename):
            input_lens = tf.reduce_sum(masks, reduction_indices=1)
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)
            out = tf.concat([fw_out, bw_out], 2)
            out = tf.nn.dropout(out, self.keep_prob)
            return out

class BiDAF(object):
    """
    Module for bidirectional attention flow.
    """

    def __init__(self, keep_prob, vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          vec_size: size of the word embeddings. int
        """
        self.keep_prob = keep_prob
        self.vec_size = vec_size
        self.S_W = tf.get_variable('S_W', [vec_size*3], tf.float32,
            tf.contrib.layers.xavier_initializer())


    def build_graph(self, q, q_mask, c, c_mask):
        """
        Inputs:
          c: context matrix, shape: (batch_size, num_context_words, vec_size).
          c_mask: Tensor shape (batch_size, num_context_words).
            1s where there's real input, 0s where there's padding
          q: question matrix (batch_size, num_question_words, vec_size)
          q_mask: Tensor shape (batch_size, num_question_words).
            1s where there's real input, 0s where there's padding
          N = num_context_words
          M = Num_question_words
          vec_size = hidden_size * 2
        Outputs:
          output: Tensor shape (batch_size, N, vec_size*3).
            This is the attention output.
        """
        with vs.variable_scope("BiDAF"):

            # Calculating similarity matrix
            c_expand = tf.expand_dims(c,2)  #[batch,N,1,2h]
            print(c_expand)
            q_expand = tf.expand_dims(q,1)  #[batch,1,M,2h]
            print(q_expand)
            c_pointWise_q = c_expand * q_expand  #[batch,N,M,2h]
            print(c_pointWise_q)

            c_input = tf.tile(c_expand, [1, 1, tf.shape(q)[1], 1])
            q_input = tf.tile(q_expand, [1, tf.shape(c)[1], 1, 1])

            concat_input = tf.concat([c_input, q_input, c_pointWise_q], -1) # [batch,N,M,6h]
            print(concat_input)


            similarity=tf.reduce_sum(concat_input * self.S_W, axis=3)  #[batch,N,M]
            print(similarity)

            # Calculating context to question attention
            similarity_mask = tf.expand_dims(q_mask, 1) # shape (batch_size, 1, M)
            print(similarity_mask)
            _, c2q_dist = masked_softmax(similarity, similarity_mask, 2) # shape (batch_size, N, M). take softmax over q
            print(c2q_dist)

            # Use attention distribution to take weighted sum of values
            c2q = tf.matmul(c2q_dist, q) # shape (batch_size, N, vec_size)
            print(c2q)

            # Calculating question to context attention c_dash
            S_max = tf.reduce_max(similarity, axis=2) # shape (batch, N)
            print(S_max)
            _, c_dash_dist = masked_softmax(S_max, c_mask, 1) # distribution of shape (batch, N)
            print(c_dash_dist)
            c_dash_dist_expand = tf.expand_dims(c_dash_dist, 1) # shape (batch, 1, N)
            print(c_dash_dist_expand)
            c_dash = tf.matmul(c_dash_dist_expand, c) # shape (batch_size, 1, vec_size)
            print(c_dash)

            c_c2q = c * c2q # shape (batch, N, vec_size)
            print(c_c2q)

            c_c_dash = c * c_dash # shape (batch, N, vec_size)
            print(c_c_dash)

            # concatenate the output
            output = tf.concat([c2q, c_c2q, c_c_dash], axis=2) # (batch_size, N, vec_size * 3)
            print(output)


            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)
            print(output)

            return output

class SimpleSoftmaxLayer:
    def __init__(self):
        pass
    def build_graph(inputs,mask):
        with vs.variable_scope(SimpleSoftmaxLayer):
            logits = tf.contrib.layers.fully_connected(inputs,num_outputs=1, activation_fn=None)
            logits = tf.squeeze(logits, axis=[2])
            masked_logits, prob_dist = masked_logits(logits,mask,1)
            return masked_logits, prob_dist

        
def masked_softmax(logits,mask,dim):
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist