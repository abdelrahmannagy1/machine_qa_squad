"""
Read tokenized data from file,truncate, pad and process it into batches ready for training

"""

from __future__ import absolute_import
from __future__ import division

import random
import time
import re

import numpy as np
from six.moves import xrange
from vocab import PAD_ID, UNK_ID

class Batch(object):
    """A class to hold the information needed for a training batch"""

    def __init__(self, context_ids, context_mask, context_tokens, qn_ids, qn_mask, qn_tokens, ans_span, ans_tokens, uuids=None):
        self.context_ids = context_ids
        self.context_mask = context_mask
        self.context_tokens = context_tokens

        self.qn_ids = qn_ids
        self.qn_mask = qn_mask
        self.qn_tokens = qn_tokens

        self.ans_span = ans_span
        self.ans_tokens = ans_tokens

        self.uuids = uuids

        self.batch_size = len(self.context_tokens)

def intstr_to_intlist(string):
    """Given a string  returns as a list of integers"""
    return [int(s) for s in string.split()]

def split_by_whitespace(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]

def sentence_to_token_ids(sentence, word2id):
    """Turns an already-tokenized sentence string into word indices
    e.g. "i do n't know" -> [9, 32, 16, 96]
    Note any token that isn't in the word2id mapping gets mapped to the id for UNK
    """
    tokens = split_by_whitespace(sentence) # list of strings
    ids = [word2id.get(w, UNK_ID) for w in tokens]
    return tokens, ids

def padded(token_batch, batch_pad=0):
    """
    Inputs:
      token_batch: List (length batch size) of lists of ints.
      batch_pad: Int. Length to pad to. If 0, pad to maximum length sequence in token_batch.
    Returns:
      List (length batch_size) of padded of lists of ints.
        All are same length - batch_pad if batch_pad!=0, otherwise the maximum length in token_batch
    """
    maxlen = max(map(lambda x: len(x), token_batch)) if batch_pad == 0 else batch_pad
    return map(lambda token_list: token_list + [PAD_ID] * (maxlen - len(token_list)), token_batch)

def refill_batches(batches, word2id, context_file, qn_file, ans_file, batch_size, context_len, question_len, discard_long):
    examples = []
    context_line, qn_line, ans_line = context_file.readline(), qn_file.readline(), ans_file.readline()
    while context_line and qn_line and ans_line:
        context_tokens, context_ids = sentence_to_token_ids(context_line, word2id)
        qn_tokens, qn_ids = sentence_to_token_ids(qn_line, word2id)
        ans_span = intstr_to_intlist(ans_line)

        context_line, qn_line, ans_line = context_file.readline(), qn_file.readline(), ans_file.readline()

        assert len(ans_span) == 2
        if ans_span[1] < ans_span[0]:
            print "Found a wrong span: start=%i end=%i" % (ans_span[0], ans_span[1])
            continue
        ans_tokens = context_tokens[ans_span[0] : ans_span[1]+1] # list of strings

        if ans_span[1] < ans_span[0]:
            print "Found an ill-formed gold span: start=%i end=%i" % (ans_span[0], ans_span[1])
            continue
        ans_tokens = context_tokens[ans_span[0] : ans_span[1]+1] # list of strings

        if len(context_ids) > context_len:
            if discard_long:
                continue
            else: # truncate
                context_ids = context_ids[:context_len]

        examples.append((context_ids, context_tokens, qn_ids, qn_tokens, ans_span, ans_tokens))

        if len(examples) == batch_size * 160:
            break

    examples = sorted(examples, key=lambda e: len(e[2]))

    for batch_start in xrange(0, len(examples), batch_size):

        # Note: each of these is a list length batch_size of lists of ints (except on last iter when it might be less than batch_size)
        context_ids_batch, context_tokens_batch, qn_ids_batch, qn_tokens_batch, ans_span_batch, ans_tokens_batch = zip(*examples[batch_start:batch_start+batch_size])

        batches.append((context_ids_batch, context_tokens_batch, qn_ids_batch, qn_tokens_batch, ans_span_batch, ans_tokens_batch))

    # shuffle the batches
    random.shuffle(batches)

    toc = time.time()
    print "Refilling batches took %.2f seconds" % (toc-tic)
    return

