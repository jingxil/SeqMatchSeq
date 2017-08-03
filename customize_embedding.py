import os
import json
import codecs
import collections
import argparse
import glob
import re

def sentence2words(s):
  # Turn words to lowercase
  s = s.lower()
  # Turn '-'  to space in sentence
  s = re.sub("[-']", ' ', s)
  # Split sentence to words
  words = s.split(' ')
  # Strip )(;"?!.,
  words = [ re.sub('[)(;"?!.,]', '', w) for w in words]
  return words

def load_embedding(embedding_path):
  vocab = []
  embedding = []
  with codecs.open(embedding_path,'r',encoding='utf8') as file:
    for line in file.readlines():
      line = line.strip()
      vocab.append(line.split(' ')[0])
      embedding.append(line)
  return vocab, embedding

def write_embedding(embedding_path,embedding):
  embedding_dir = os.path.dirname(embedding_path)
  embedding_file = os.path.basename(embedding_path)
  trimmed_embedding_path = embedding_dir+"/"+"customized_"+embedding_file
  # Write embedding
  with codecs.open(trimmed_embedding_path,'w',encoding='utf8') as file:
    embedding = '\n'.join(embedding)
    file.write(embedding)
 

def collect_data_vocab(data_path):
  words = []
  with open(data_path,'r') as file:
    for line in file.readlines():
      data = json.loads(line.strip())
      sentence1 = sentence2words(data['sentence1'])
      sentence2 = sentence2words(data['sentence2'])
      words += sentence1+sentence2
  return words


if __name__ == '__main__':
  # Create Parser
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--data_dir",
      type=str,
      default="./data",
      help="Directory to load data"
  )
  parser.add_argument(
      "--embedding_path",
      type=str,
      default="./vocab",
      help="Path to load pre-trained embedding and write back trimmed embedding"
  )
  FLAGS, _ = parser.parse_known_args()
  # Load embeding
  print("Loading pre-trained embedding")
  vocab, embedding = load_embedding(FLAGS.embedding_path)
  print("Loaded pre-trained embedding")
  # Find all files
  file_names = glob.glob(FLAGS.data_dir)
  # Get all unique words in data
  words = []
  for file_name in file_names:
    words += collect_data_vocab(file_name)
  words = [w[0] for w in collections.Counter(words).most_common()]
  print("Num of unique words: %d" % (len(words)))
  # Keep the part of embedding which we need
  print("Start to chop embedding")

  trimmed_embedding = []
  word2id = {}
  for v in vocab:
    word2id[v]=len(word2id)
  for word in words:
    if word in vocab:
      trimmed_embedding.append(embedding[word2id[word]])

  print("num of OOV words: %d" % (len(words)-len(trimmed_embedding)))
  print("Finished chopping embedding")   
  # Write back trimmed embedding
  write_embedding(FLAGS.embedding_path,trimmed_embedding)

