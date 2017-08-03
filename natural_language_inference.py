import tensorflow as tf
import numpy as np
import time
import os
import json
import codecs
import queue 
from random import shuffle
from tensorflow.python import debug as tf_debug
import seq_match_seq
from customize_embedding import sentence2words

DEBUG = False


tf.app.flags.DEFINE_integer("batch_size", 64,"Batch size.")
tf.app.flags.DEFINE_integer("max_premise_len", 78, "Maximum premise sequence length.")
tf.app.flags.DEFINE_integer("max_hypothesis_len", 59, "Maximum hypothesis sequence length.")
tf.app.flags.DEFINE_integer("rnn_size", 256, "RNN unit size.")
tf.app.flags.DEFINE_integer("attention_size", 256, "Attention size.")
tf.app.flags.DEFINE_float("dropout_rate", 0.3, "Dropout rate.")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_integer("decay_epochs", 15, "Number of epochs model keeps learning rate no change.")
tf.app.flags.DEFINE_integer("num_epochs", 15, "Number of epochs model runs.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate decay factor.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Maximum gradient norm.")
tf.app.flags.DEFINE_boolean("forward_only", False, "Forward Only.")
tf.app.flags.DEFINE_string("log_dir", "./log", "Log directory")
tf.app.flags.DEFINE_string("data_path", "./snli_1.0/snli_1.0_train.jsonl", "Data path.")
tf.app.flags.DEFINE_string("embedding_path", "./vocab/glove.6B.200d.txt", "Embedding path.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "Frequence to do per checkpoint.")

FLAGS = tf.app.flags.FLAGS


class NLI(object):
  def __init__(self, forward_only):
    self._forward_only = forward_only

    self._target_table = {'neutral':0,'entailment':1,'contradiction':2,'-':3}
    self._queue = queue.Queue()
    self._num_epoches = 0

    self.load_embedding()
    self.read_data()
    self._graph = tf.Graph()
    with self._graph.as_default():
      self._sess = tf.Session()
      if DEBUG:
        self._sess = tf_debug.LocalCLIDebugWrapperSession(self._sess)
        self._sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)  
    self.build_model()
  
  def load_embedding(self):
    """ Load pre-trained embedding
        Prepare vocab, word2id
    """
    vocab = []
    embedding = []
    with codecs.open(FLAGS.embedding_path,'r',encoding='utf8') as file:
      for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embedding.append([ float(i) for i in row[1:]])
    self._embedding = embedding
    self._embedding_dimension = len(embedding[0])
    self._word2id = {'<UNK>':seq_match_seq.UNK_ID}
    for v in vocab:
      self._word2id[v] = len(self._word2id)
    self._vocab = self._word2id.keys()
    
  def read_data(self):    
    records = []
    with open(FLAGS.data_path,'r') as file:
      for line in file.readlines():
        data = json.loads(line.strip())
        target = data['gold_label']
        # Ignore '-' type
        if target == '-':
          continue
        sentence1 = sentence2words(data['sentence1'])
        sentence2 = sentence2words(data['sentence2'])
        sentence1 = [self._word2id[w] if w in self._word2id else seq_match_seq.UNK_ID  for w in sentence1]
        sentence2 = [self._word2id[w] if w in self._word2id else seq_match_seq.UNK_ID for w in sentence2]
        rec = {}
        rec['premise']= sentence1
        rec['hypothesis']= sentence2
        rec['target']= self._target_table[target]
        rec['premise_len']= len(sentence1)
        rec['hypothesis_len']= len(sentence2)
        records.append(rec)
    self._data = records
    self._data_size = len(records)
    # Statistic
    premise_lens = [ rec['premise_len'] for rec in records]
    hypothesis_lens = [ rec['hypothesis_len'] for rec in records]
    print("Data size: %d" % (len(records)))
    print("max_premise_len: %d max_hypothesis_len: %d" % (max(premise_lens),max(hypothesis_lens)))
    print("*Snippet*")
    print("premise: "+str(records[0]['premise']))
    print("hypothesis: "+str(records[0]['hypothesis']))
    print("target: "+str(records[0]['target']))
    print("premise_len: "+str(records[0]['premise_len']))
    print("hypothesis_len: "+str(records[0]['hypothesis_len']))

  def fill_queue(self):
    order = list(range(len(self._data)))
    shuffle(order)
    for i in order:
      self._queue.put_nowait(i)

  def get_batch(self):
    premises = []
    hypothesises = []
    premise_lens = []
    hypothesis_lens = []
    targets = []

    while len(targets) < FLAGS.batch_size:
      try:
        idx = self._queue.get_nowait()
        rec = self._data[idx]
        premises.append(rec['premise'])
        hypothesises.append(rec['hypothesis'])
        premise_lens.append(rec['premise_len'])
        hypothesis_lens.append(rec['hypothesis_len'])
        targets.append(rec['target'])
      except queue.Empty as e:
        self.fill_queue()
    # Pad every sequence to the same length    
    max_premise_len = max(premise_lens)
    max_hypothesis_len = max(hypothesis_lens)
    for i in range(len(premises)):
      premises[i] = premises[i] + [0]*(max_premise_len-len(premises[i]))
    for i in range(len(hypothesises)):
      hypothesises[i] = hypothesises[i] + [0]*(max_hypothesis_len-len(hypothesises[i]))
    # Split premises and hypothesises to list
    time_major_premises = np.split(np.array(premises),max_premise_len,1)
    time_major_hypothesises = np.split(np.array(hypothesises),max_hypothesis_len,1)
    # Reshape
    time_major_premises = [ np.reshape(i,(-1)) for i in time_major_premises]
    time_major_hypothesises = [ np.reshape(i,(-1)) for i in time_major_hypothesises]

    return time_major_premises,time_major_hypothesises,premise_lens,hypothesis_lens,targets

  def build_model(self):
    with self._graph.as_default():
      # Build model
      self._model = seq_match_seq.SeqMatchSeq(batch_size=FLAGS.batch_size,
                                        rnn_size=FLAGS.rnn_size,
                                        attention_size=FLAGS.attention_size,
                                        dropout_rate=FLAGS.dropout_rate,
                                        max_premise_len=FLAGS.max_premise_len,
                                        max_hypothesis_len=FLAGS.max_hypothesis_len,
                                        embedding=self._embedding, 
                                        embedding_dimension=self._embedding_dimension,
                                        learning_rate=FLAGS.learning_rate,
                                        learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
                                        max_gradient_norm=FLAGS.max_gradient_norm,
                                        forward_only=FLAGS.forward_only)
      
      if FLAGS.forward_only == False:
        # Prepare Summary writer
        self._writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',self._sess.graph)
      
      # Try to get checkpoint
      ckpt = tf.train.get_checkpoint_state(FLAGS.log_dir)
      if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Load model parameters from %s" % ckpt.model_checkpoint_path)
        self._model.saver.restore(self._sess, ckpt.model_checkpoint_path)
      else:
        print("Created model with fresh parameters.")
        self._sess.run(tf.global_variables_initializer())

  def train(self):
    loss = 0.0
    step_time = 0.0
    current_step = 0
    previous_losses = []
    while True:
      start_time = time.time()
      time_major_premises,time_major_hypothesises,premise_lens,hypothesis_lens,targets = self.get_batch()

      summary, step_loss, predicted_ids_with_logits, debug_var = \
                  self._model.step(self._sess,time_major_premises,premise_lens,time_major_hypothesises,hypothesis_lens,targets)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # DEBUG PART
      #print("debug")
      #print(debug_var)
      #return
      # /DEBUG PART

      # Time to print statistic and save model
      if current_step % FLAGS.steps_per_checkpoint == 0:
        with self._sess.as_default():
          gstep = self._model.global_step.eval()
          lr = self._model.learning_rate.eval()
        # Compute Epoch
        self._num_epoches = gstep*FLAGS.batch_size//self._data_size  
        print ("epoch %d global step %d learning rate %f step-time %.2f loss %.2f" % (self._num_epoches, gstep, lr, step_time, loss))
        # after FLAGS.decay_epochs, decrease learning rate if loss does not decline compared with previous 3 intervals 
        if self._num_epoches > FLAGS.decay_epochs:
          if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
            previous_losses = previous_losses[-3:]
            self._sess.run(self._model.learning_rate_decay_op)
        previous_losses.append(loss)

        # Write summary
        self._writer.add_summary(summary, gstep)
        # Batch correct rate
        predicted_ids = np.reshape(predicted_ids_with_logits[1],(-1))
        batch_correct_count = np.sum(np.equal(targets,predicted_ids))
        print("="*20)
        print("Batch correct rate: %f" % (batch_correct_count/FLAGS.batch_size))
        print("="*20)  
        checkpoint_path = os.path.join(FLAGS.log_dir, "nli.ckpt")
        self._model.saver.save(self._sess, checkpoint_path, global_step=self._model.global_step)
        step_time, loss = 0.0, 0.0

        # Time to stop
        if self._num_epoches > FLAGS.num_epochs:
          return


  def eval(self):
    sample_count = 0
    correct_count = 0

    batch_count = self._data_size // FLAGS.batch_size
    remaining_size = self._data_size % FLAGS.batch_size

    for i in range(batch_count+1):
      time_major_premises,time_major_hypothesises,premise_lens,hypothesis_lens,targets = self.get_batch()
      predicted_ids_with_prob = self._model.step(self._sess,time_major_premises,premise_lens,time_major_hypothesises,hypothesis_lens)
      predicted_ids = np.reshape(predicted_ids_with_prob[1],(-1))
      rv = np.equal(targets,predicted_ids)
      if i == batch_count:
        rv = rv[:remaining_size]
      batch_correct_count = np.sum(rv)
      sample_count += len(rv)
      correct_count += batch_correct_count
      print("Batch correct rate: %f" % (batch_correct_count/FLAGS.batch_size))
    print("Overall Result:\n total number of samples: %d \n number of correct predictions %d \n correct rate: %f" 
          % (sample_count,correct_count, correct_count/sample_count))

  def run(self):
    if self._forward_only:
      self.eval()
    else:
      self.train()

def main(_):
  nli = NLI(FLAGS.forward_only)
  nli.run()


if __name__ == "__main__":
  tf.app.run() 