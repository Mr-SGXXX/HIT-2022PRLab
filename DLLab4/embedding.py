# 用于评论向量化表示的词嵌入生成、加载
import os
import csv
import jieba
import torch
import torch.nn as nn
from gensim.models import KeyedVectors, Word2Vec
import multiprocessing


def gen_embedding(data_path, embedding_path, embedding_size):
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        sentences = []
        for row in reader:
            sentence = jieba.lcut(row['review'])
            sentences.append(sentence)
        model = Word2Vec(sentences, vector_size=embedding_size, workers=multiprocessing.cpu_count())
        model.wv.save_word2vec_format(embedding_path, binary=True)


def load_embedding(data_path, embedding_path='embedding.bin', embedding_size=100, logger=None):
    if not os.path.exists(embedding_path):
        gen_embedding(data_path, embedding_path=embedding_path, embedding_size=embedding_size)
        if logger is not None:
            logger.info("Embedding Generated")
    elif logger is not None:
        logger.info("Embedding Found")
    try:
        wvmodel = KeyedVectors.load_word2vec_format(embedding_path, binary=True, unicode_errors='ignore')
        idx_to_word = wvmodel.index_to_key
        word_to_idx = {wvmodel.index_to_key[idx]: idx + 1 for idx in range(len(idx_to_word))}
        word_to_idx["unk"] = 0
        weight = torch.zeros(len(idx_to_word) + 1, embedding_size)
        for i in range(len(idx_to_word)):
            weight[i + 1, :] = torch.from_numpy(wvmodel.get_vector(wvmodel.index_to_key[i]))
        embedding = nn.Embedding.from_pretrained(weight)
        if logger is not None:
            logger.info("Embedding Load Over")
        return embedding, word_to_idx
    except:
        if logger is not None:
            logger.info("Embedding Load Error And Retry")
        os.remove(embedding_path)
        return load_embedding(data_path, embedding_path, embedding_size)
