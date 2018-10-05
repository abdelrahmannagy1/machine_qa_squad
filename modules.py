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

        
