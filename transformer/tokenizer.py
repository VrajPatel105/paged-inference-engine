import re 
class Tokenizer():

  PAD_ID = 0
  UNK_ID = 1
  SOS_ID = 2
  EOS_ID = 3

  def __init__(self):

    self.word2idx = {
        "[PAD]": self.PAD_ID,
        "[UNK]": self.UNK_ID,
        "[SOS]": self.SOS_ID,
        "[EOS]": self.EOS_ID
    }

    self.idx2word = {
        self.PAD_ID: "[PAD]",
        self.UNK_ID: "[UNK]",
        self.SOS_ID: "[SOS]",
        self.EOS_ID: "[EOS]"
    }

    self.next_id = 4

  def preprocess(self, sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return sentence.split()

  def build_vocab(self, sentences):
    for sentence in sentences:
      words = self.preprocess(sentence)
      for word in words:
        if word not in self.word2idx:
          self.word2idx[word] = self.next_id
          self.idx2word[self.next_id] = word
          self.next_id = self.next_id + 1

  def encode_sentence(self, sentence, add_sos=False, add_eos=False):
    words = self.preprocess(sentence)
    encoded_list = []
    for word in words:
        encoded_list.append(self.word2idx.get(word, self.UNK_ID))
    if add_sos:
        encoded_list.insert(0, self.SOS_ID)
    if add_eos:
        encoded_list.append(self.EOS_ID)
    return encoded_list

  def decode_sentence(self, ids):
    words = [self.idx2word.get(i, "[UNK]") for i in ids]
    return " ".join(words)

  def vocab_size(self):
    return len(self.word2idx)
