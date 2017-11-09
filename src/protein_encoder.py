#adapted from tensor2tensor models

"""Encoders for Protein data.

* ProteinEncoder: AA strings to ints and back
* DelimitedProteinEncoder: for delimited subsequences
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
#import subprocess
# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import problem
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer

class ProteinEncoder(text_encoder.TextEncoder):
  """AA (amino acid) strings to ints and back.

  Uses 'X' as an unknown AA.
  """
  AAS = list("ACDEFGHIKLMNPQRSTVWY")
  AAS += list("UOBZJ") #uncommon or confusing AAs, see http://wiki.thegpm.org/wiki/Amino_acid_symbols
  UNK = "X" #unknown AA
  PAD = "0"

  def __init__(self,
               chunk_size=1,
               num_reserved_ids=text_encoder.NUM_RESERVED_TOKENS):
    super(ProteinEncoder, self).__init__(num_reserved_ids=num_reserved_ids)
    # Build a vocabulary of chunks of size chunk_size
    self._chunk_size = chunk_size
    tokens = self._tokens()
    tokens.sort()
    ids = range(self._num_reserved_ids, len(tokens) + self._num_reserved_ids)
    self._ids_to_tokens = dict(zip(ids, tokens))
    self._tokens_to_ids = dict(zip(tokens, ids))

  def _tokens(self):
    chunks = []
    for size in range(1, self._chunk_size + 1):
      c = itertools.product(self.AAS + [self.UNK], repeat=size)
      num_pad = self._chunk_size - size
      padding = (self.PAD,) * num_pad
      c = [el + padding for el in c]
      chunks.extend(c)
    return chunks

  @property
  def vocab_size(self):
    return len(self._ids_to_tokens) + self._num_reserved_ids

  def encode(self, s):
    aas = list(s)
    extra = len(aas) % self._chunk_size
    if extra > 0:
      pad = [self.PAD] * (self._chunk_size - extra)
      aas.extend(pad)
    assert (len(aas) % self._chunk_size) == 0
    num_chunks = len(aas) // self._chunk_size
    ids = []
    for chunk_idx in xrange(num_chunks):
      start_idx = chunk_idx * self._chunk_size
      end_idx = start_idx + self._chunk_size
      chunk = tuple(aas[start_idx:end_idx])
      if chunk not in self._tokens_to_ids:
        raise ValueError("Unrecognized token %s" % chunk)
      ids.append(self._tokens_to_ids[chunk])
    '''for start_idx in xrange(len(aas)-self._chunk_size+1):
      chunk = tuple(aas[start_idx:start_idx+self._chunk_size])
      if chunk not in self._tokens_to_ids:
        raise ValueError("Unrecognized token %s" % chunk)
      ids.append(self._tokens_to_ids[chunk])'''
    return ids

  def decode(self, ids):
    aas = []
    for idx in ids:
      if idx >= self._num_reserved_ids:
        chunk = self._ids_to_tokens[idx]
        if self.PAD in chunk:
          chunk = chunk[:chunk.index(self.PAD)]
      else:
        chunk = [text_encoder.RESERVED_TOKENS[idx]]
      aas.extend(chunk)
    '''for index in range(0, self._chunk_size, len(ids)-1):
      idx = idx[index]
        if idx >= self._num_reserved_ids:
          chunk = self._ids_to_tokens[idx]
          chunk = chunk[:1]
        else:
          chunk = [text_encoder.RESERVED_TOKENS[idx]]
      aas.extend(chunk)'''
    return "".join(aas)


class DelimitedProteinEncoder(ProteinEncoder):
  """ProteinEncoder for delimiter separated subsequences.

  Uses '\n' as default delimiter.
  """

  def __init__(self, delimiter="\n", **kwargs):
    self._delimiter = delimiter
    super(DelimitedProteinEncoder, self).__init__(**kwargs)

  @property
  def delimiter(self):
    return self._delimiter

  def _tokens(self):
    return super(DelimitedProteinEncoder, self)._tokens() + [self.delimiter]

  def encode(self, delimited_string):
    ids = []
    for s in delimited_string.split(self.delimiter):
      ids.extend(super(DelimitedProteinEncoder, self).encode(s))
      ids.append(self._tokens_to_ids[self.delimiter])
    return ids[:-1]

@registry.register_problem
class ProteinEmbeddingProblem(problem.Text2TextProblem):
  
  @property
  def is_character_level(self):
    return True

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_CHR

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_CHR

  @property
  def vocab_name(self):
    return "vocab.proteinembed"

  @property
  def num_shards(self):
    return 100

  def generator(self, data_dir, tmp_dir, train):
    seq_filepath = data_dir+"/homo_sapiens_prot2vec_sequences.txt"
    aa_encoder = ProteinEncoder()
    with open(seq_filepath, 'r') as seq_file:
      seq = seq_file.readline().strip()
      while seq:
        seq_encoded = aa_encoder.encode(seq)
        yield {"inputs":seq_encoded, "targets":seq_encoded}
        seq = seq_file.readline().strip()

@registry.register_hparams
def transformer_protein2vec():
  hparams = transformer.transformer_base()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 128
  hparams.filter_size = 512
  hparams.num_heads = 4
  hparams.batch_size = 1024
  return hparams

#python t2t-trainer --t2t_usr_dir=../../../../wyss/church/protein2vec/src --registry_help
#python t2t-trainer --generate_data --t2t_usr_dir=../../../../wyss/church/protein2vec/src --data_dir=../../../../wyss/church/protein2vec/dat --problems=protein_embedding_problem --model=transformer --hparams_set=transformer_base_single_gpu --output_dir=../../../../wyss/church/protein2vec/out
#python t2t-trainer --t2t_usr_dir=../../../../wyss/church/protein2vec/src --data_dir=../../../../wyss/church/protein2vec/dat --problems=protein_embedding_problem --model=transformer --hparams_set=transformer_tiny --hparams='batch_size=1024' --output_dir=../../../../wyss/church/protein2vec/out
#python t2t-trainer --t2t_usr_dir=../../../../wyss/church/protein2vec/src --data_dir=../../../../wyss/church/protein2vec/dat --problems=protein_embedding_problem --model=transformer --hparams_set=transformer_protein2vec --output_dir=../../../../wyss/church/protein2vec/out
