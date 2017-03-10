import tensorflow as tf
import codecs
import os
import jieba
import collections
import re

def readfile(file_path):
    f = codecs.open(file_path, 'r', 'utf-8')
    alltext = f.read()
    alltext = re.sub(r'\s','', alltext)
    seglist = list(jieba.cut(alltext, cut_all = False))
    return seglist
    
def _build_vocab(filename):
    data = readfile(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict(zip(range(len(words)),words))
    dataids = []
    for w in data:
        dataids.append(word_to_id[w])
    return word_to_id, id_to_word,dataids

def dataproducer(batch_size, num_steps):
    word_to_id, id_to_word, data = _build_vocab('F:\\ml\\code\\lstm\\1.txt')
    datalen = len(data)
    batchlen = datalen//batch_size
    epcho_size = (batchlen - 1)//num_steps

    data = tf.reshape(data[0: batchlen*batch_size], [batch_size,batchlen])
    i = tf.train.range_input_producer(epcho_size, shuffle=False).dequeue()
    x = tf.slice(data, [0,i*num_steps],[batch_size, num_steps])
    y = tf.slice(data, [0,i*num_steps+1],[batch_size, num_steps])
    x.set_shape([batch_size, num_steps])
    y.set_shape([batch_size, num_steps])
    return x,y,id_to_word
    
    
    
    