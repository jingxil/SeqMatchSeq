import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.layers import core as layers_core
import collections


UNK_ID = 0

def get_hidden_state(cell_state):
  """ Get the hidden state needed in cell state which is 
      possibly returned by LSTMCell, GRUCell, RNNCell or MultiRNNCell.
  
  Args:
    cell_state: a structure of cell state

  Returns:
    hidden_state: A Tensor
  """

  if type(cell_state) is tuple:
    cell_state = cell_state[-1]
  if hasattr(cell_state, "h"):
    hidden_state = cell_state.h
  else:
    hidden_state = cell_state
  return hidden_state  

class SeqMatchSeqAttentionState(
    collections.namedtuple("SeqMatchSeqAttentionState", ("cell_state", "attention"))):
  pass

class SeqMatchSeqAttention(object):
  """ Attention for SeqMatchSeq.
  """

  def __init__(self,num_units,premise_mem,premise_mem_weights,name="SeqMatchSeqAttention"):
    """ Init SeqMatchSeqAttention

    Args:
      num_units: The depth of the attention mechanism.
      premise_mem: encoded premise memory
      premise_mem_weights: premise memory weights
    """
    # Init layers
    self._name = name
    self._num_units = num_units
    # Shape: [batch_size,max_premise_len,rnn_size]
    self._premise_mem = premise_mem
    # Shape: [batch_size,max_premise_len]
    self._premise_mem_weights = premise_mem_weights

    with tf.name_scope(self._name):
      self.query_layer = layers_core.Dense(num_units, name="query_layer", use_bias=False)
      self.hypothesis_mem_layer = layers_core.Dense(num_units, name="hypothesis_mem_layer", use_bias=False)
      self.premise_mem_layer = layers_core.Dense(num_units, name="premise_mem_layer", use_bias=False)
      # Preprocess premise Memory
      # Shape: [batch_size, max_premise_len, num_units]
      self._keys = self.premise_mem_layer(premise_mem)
      self.batch_size = self._keys.shape[0].value 
      self.alignments_size = self._keys.shape[1].value 

  def __call__(self, hypothesis_mem, query):
    """ Perform attention

    Args:
      hypothesis_mem: hypothesis memory
      query: hidden state from last time step

    Returns:
      attention: computed attention
    """
    with tf.name_scope(self._name):
      # Shape: [batch_size, 1, num_units]
      processed_hypothesis_mem = tf.expand_dims(self.hypothesis_mem_layer(hypothesis_mem), 1)
      # Shape: [batch_size, 1, num_units]
      processed_query = tf.expand_dims(self.query_layer(query), 1)
      v = tf.get_variable("attention_v", [self._num_units], dtype=tf.float32)
      # Shape: [batch_size, max_premise_len]
      score = tf.reduce_sum(v * tf.tanh(self._keys + processed_hypothesis_mem + processed_query), [2])
      # Mask score with -inf
      score_mask_values = float("-inf") * (1.-tf.cast(self._premise_mem_weights, tf.float32))
      masked_score = tf.where(tf.cast(self._premise_mem_weights, tf.bool), score, score_mask_values)
      # Calculate alignments
      # Shape: [batch_size, max_premise_len]
      alignments = tf.nn.softmax(masked_score)
      # Calculate attention
      # Shape: [batch_size, rnn_size]
      attention = tf.reduce_sum(tf.expand_dims(alignments, 2) * self._premise_mem, axis=1)
      return attention


class SeqMatchSeqWrapper(rnn_cell_impl.RNNCell):
  """ RNN Wrapper for SeqMatchSeq.
  """
  def __init__(self, cell, attention_mechanism, name='SeqMatchSeqWrapper'):
    super(SeqMatchSeqWrapper, self).__init__(name=name)
    self._cell = cell
    self._attention_mechanism = attention_mechanism

  def call(self, inputs, state):
    """
    Args:
      inputs: inputs at some time step
      state: A (structure of) cell state
    """
    # Concatenate attention and input 
    cell_inputs = tf.concat([state.attention, inputs], axis=-1)
    cell_state = state.cell_state
    # Call cell function
    cell_output, next_cell_state = self._cell(cell_inputs, cell_state)
    # Get hidden state
    hidden_state = get_hidden_state(cell_state)
    # Calculate attention
    attention = self._attention_mechanism(inputs, hidden_state)
    # Assemble next state
    next_state = SeqMatchSeqAttentionState(
      cell_state=next_cell_state,
      attention=attention)
    return cell_output, next_state

  @property
  def state_size(self):
    return SeqMatchSeqAttentionState(
        cell_state=self._cell.state_size,
        attention=self._attention_mechanism._premise_mem.get_shape()[-1].value
        )

  @property
  def output_size(self):
    return self._cell.output_size

  def zero_state(self, batch_size, dtype):
    cell_state = self._cell.zero_state(batch_size, dtype)
    attention = rnn_cell_impl._zero_state_tensors(self.state_size.attention, batch_size, tf.float32)
    return SeqMatchSeqAttentionState(
          cell_state=cell_state,
          attention=attention)


class SeqMatchSeq(object):
  """ Sequence Match Sequence Model
  
  This class implements the Sequence Match Sequence Model
  as described in this paper: http://arxiv.org/pdf/1512.08849v1.pdf
  except
    1. Here I use <UNK> represents unknown words whereas in the papar 
       unknown words is represented by average over surroundding words.
    2. dropout is supported
  """

  def __init__(self,batch_size,rnn_size,attention_size,dropout_rate,max_premise_len,max_hypothesis_len,embedding,embedding_dimension,
                learning_rate,learning_rate_decay_factor,max_gradient_norm,forward_only=False):
    """Create the model.
    Args:
      batch_size: Batch size
      rnn_size: Size of RNN hidden units
      attention_size: Size of attention mechanism
      dropout_rate: The percentage of features are discarded
      max_premise_len: Maximum premise length
      max_hypothesis_len: Maximum hypothesis length
      embedding: Pre-trained word embeddings
      embedding_dimension: The dimensionality of word embeddings
      learning_rate: Learning rate
      learning_rate_decay_factor: Learning rate decay factor
      max_gradient_norm: Maximum gradient norm
      forward_only: Whether forward only
    """

    # Create global step
    self._forward_only = forward_only
    self.global_step = tf.get_variable('global_step',shape=[],initializer=tf.constant_initializer(0,dtype=tf.int32),trainable=False)
    self.learning_rate = tf.get_variable('learning_rate',shape=[],initializer=tf.constant_initializer(learning_rate,dtype=tf.float32),trainable=False)
    self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
    # UNK token's embedding
    special_token_embedding = tf.get_variable("special_token_embedding", [1, embedding_dimension],dtype=tf.float32)
    # Load pre_trained_embedding
    pre_trained_embedding = tf.constant(embedding, name="pre_trained_embedding")
    # Concatenate embeddings
    embedding = tf.concat([special_token_embedding,pre_trained_embedding],0)
    # Create placeholders
    self._premise = []
    self._hypothesis = []
    for i in range(max_premise_len):
      self._premise.append(tf.placeholder_with_default(tf.constant([0]*batch_size,dtype=tf.int32), shape=[batch_size],name="premise_t{0}".format(i)))
    for i in range(max_hypothesis_len):
      self._hypothesis.append(tf.placeholder_with_default(tf.constant([0]*batch_size,dtype=tf.int32), shape=[batch_size],name="hypothesis_t{0}".format(i)))
    self._premise_lens = tf.placeholder(tf.int32, shape=[batch_size], name="premise_lens")
    self._hypothesis_lens = tf.placeholder(tf.int32, shape=[batch_size], name="hypothesis_lens")
    self._targets = tf.placeholder(tf.int32, shape=[batch_size], name="targets")
    # Calculate sequence masks
    premise_weights = tf.cast(tf.sequence_mask(self._premise_lens, max_premise_len),tf.int32)
    hypothesis_weights = tf.cast(tf.sequence_mask(self._hypothesis_lens, max_hypothesis_len),tf.int32)
    # Stack premise and hypothesis
    premise = tf.stack(self._premise,1)
    hypothesis = tf.stack(self._hypothesis,1)
    # Embed
    embedded_premise = tf.nn.embedding_lookup(embedding,premise)
    embedded_hypothesis = tf.nn.embedding_lookup(embedding,hypothesis)
    # Choose RNN Cell
    cell = tf.contrib.rnn.LSTMCell
    with tf.variable_scope("premise_encoding"):
      # Create premise encoder with dropout
      premise_encoder = tf.contrib.rnn.DropoutWrapper(cell(rnn_size),input_keep_prob=1-dropout_rate, output_keep_prob=1-dropout_rate)
      # Encode premise
      # Shape: [batch_size, max_time, rnn_size]
      premise_mem,_ = tf.nn.dynamic_rnn(premise_encoder,embedded_premise,self._premise_lens,dtype=tf.float32)
    with tf.variable_scope("hypothesis_encoding"):
      # Create hypothesis encoder with dropout
      hypothesis_encoder = tf.contrib.rnn.DropoutWrapper(cell(rnn_size),input_keep_prob=1-dropout_rate, output_keep_prob=1-dropout_rate)
      # Encode hypothesis
      # Shape: [batch_size, max_time, rnn_size]
      hypothesis_mem,_ = tf.nn.dynamic_rnn(hypothesis_encoder,embedded_hypothesis,self._hypothesis_lens,dtype=tf.float32)
    # Use SeqMatchSeq Attention Mechanism
    attention_mechanism = SeqMatchSeqAttention(attention_size, premise_mem, premise_weights)
    # match LSTM
    mLSTM = cell(rnn_size)
    # Wrap mLSTM
    mLSTM = SeqMatchSeqWrapper(mLSTM,attention_mechanism)

    # Training Helper
    #helper = tf.contrib.seq2seq.TrainingHelper(hypothesis_mem, self._hypothesis_lens)    
    # Basic Decoder
    #decoder = tf.contrib.seq2seq.BasicDecoder(mLSTM, helper, mLSTM.zero_state(batch_size,tf.float32)) 
    # Decode
    #_, state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,impute_finished=True)

    _, state = tf.nn.dynamic_rnn(mLSTM, hypothesis_mem, self._hypothesis_lens, dtype=tf.float32)
    hidden_state = get_hidden_state(state.cell_state)
    # Fully connection Layer
    fcn = layers_core.Dense(3, name='fcn')
    # logits
    logits = fcn(hidden_state)
    if self._forward_only:
      prob = tf.nn.softmax(logits)
      self._predicted_ids_with_prob = tf.nn.top_k(prob) 
    else:
      # predicted_ids_with_logits
      self._predicted_ids_with_logits=tf.nn.top_k(logits)
      # Losses
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._targets, logits=logits)
      # Total loss
      self._loss = tf.reduce_sum(losses)/batch_size
      # Get all trainable variables
      parameters = tf.trainable_variables()
      # Calculate gradients
      gradients = tf.gradients(self._loss, parameters)
      # Clip gradients
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
      # Optimization
      #optimizer = tf.train.GradientDescentOptimizer(self.init_learning_rate)
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      # Update operator
      self._update_op = optimizer.apply_gradients(zip(clipped_gradients, parameters),global_step=self.global_step)
      # Summarize
      tf.summary.scalar('learning_rate',self.learning_rate)
      tf.summary.scalar('loss',self._loss)
      for p in parameters:
        tf.summary.histogram(p.op.name,p)
      for p in gradients:
        tf.summary.histogram(p.op.name,p)
      # Summarize
      self._summary = tf.summary.merge_all()
      #DEBUG PART
      self._debug_var = self._loss
      #/DEBUG PART

    # Saver
    self.saver = tf.train.Saver(tf.global_variables())

  def step(self, session, premise, premise_lens, hypothesis, hypothesis_lens, targets=None):
    """Run a step of the model feeding the given inputs.
    
    Returns:
      (predicting)
      The predicted ids with probabilities

      (training)
      The summary      
      The total loss
      The predicted ids with logits
      The variable for debugging

    """
    #Fill up inputs 
    input_feed = {}
    for i in range(len(premise)):
      input_feed[self._premise[i]] = premise[i]
    for i in range(len(hypothesis)):
      input_feed[self._hypothesis[i]] = hypothesis[i] 
    input_feed[self._premise_lens] = premise_lens
    input_feed[self._hypothesis_lens] = hypothesis_lens
    if self._forward_only==False:
      input_feed[self._targets] = targets

    #Fill up outputs
    if self._forward_only:
      output_feed = [self._predicted_ids_with_prob]
    else:
      output_feed = [self._update_op, self._summary, self._loss, self._predicted_ids_with_logits, self._debug_var]

    #Run step
    outputs = session.run(output_feed, input_feed)


    #Return
    if self._forward_only:
      return outputs[0] 
    else:
      return outputs[1],outputs[2],outputs[3],outputs[4]