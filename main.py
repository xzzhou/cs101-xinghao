#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:28:35 2018

@author: Xinghao
"""

from read import *
from constants import *
from train import *
from helper import *
from model import *
import datetime

import random


input_lang, output_lang, pairs = prepareData('diag1', 'diag2-other-omit2', False)
print(random.choice(pairs))


encoder1 = EncoderRNN(input_lang.n_words, hidden_size)

attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1)

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

trainIters(encoder1, attn_decoder1, input_lang, output_lang, pairs, 200000, print_every=5000)

now = str(datetime.datetime.now())[:16].replace('-','_').replace(':', '_').replace(' ', '_')
save('./savedModel/two_sentences_' + now, encoder1, attn_decoder1)



