#!/usr/bin/python
#-*-coding:utf-8 -*-
#  Copyright (c) <2014>, <Mohamed Sordo>
# Email: mohamed ^dot^ sordo ^at^ gmail ^dot^ com
# Website: http://msordo.weebly.com
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer. 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies, 
# either expressed or implied, of the FreeBSD Project.

import os, sys, pickle, argparse
import numpy
from gensim import corpora, models, similarities
#gensim是一个自然语言处理库，能够将文档根据TF-IDF, LDA, LSI等模型转化成向量模式，还可实现word2vec功能，能够将单词转化为词向量。
#相关用法1：http://blog.csdn.net/u014595019/article/details/52218249
#相关用法2：http://www.52nlp.cn/%E5%A6%82%E4%BD%95%E8%AE%A1%E7%AE%97%E4%B8%A4%E4%B8%AA%E6%96%87%E6%A1%A3%E7%9A%84%E7%9B%B8%E4%BC%BC%E5%BA%A6%E4%BA%8C
#相关用法3：http://blog.csdn.net/mebiuw/article/details/53870778
#相关用法4：http://blog.csdn.net/mebiuw/article/details/53870778
#原理解释1：http://www.cnblogs.com/LeftNotEasy/archive/2011/01/19/svd-and-applications.html
#原理解释2：http://blog.csdn.net/u011450885/article/details/46500901

import config, cleaner

def cosine(v1, v2):
    cos = numpy.dot(v1, v2)       #两向量点乘
    cos /= numpy.linalg.norm(v1)  #/向量长
    cos /= numpy.linalg.norm(v2)
    return cos                    #得到两向量的cos值

class MyCorpus(object):
    def __init__(self, corpus_filename, dictionary):  #初始化语料库
        self.dictionary = dictionary
        self.corpus_filename = corpus_filename
        self.docs = []
    def __iter__(self):                               #迭代器
        for line in open(self.corpus_filename):       #每行都产生文档向量
            #assume there's one document per line, items separated by tab
            doc, documents_weights = line.strip().split("\t")
            text = []
            for document, weight in eval(documents_weights):  #eval将 字符串数字 转成 数字进行求值
                text.extend([document]*int(weight))   #text增加列表参数，加入一列document，一列weight
            self.docs.append(doc)                     #增加对象参数，加入每个行第一个参数到列表中
            yield self.dictionary.doc2bow(text)       #yield生成器，doc2bow变为词袋，输出格式是[ [(0,1)(1,1)(2,1)] [(0,1)(4,1)(9,2)] ]
                                                      #将 用字符串表示的文档 转换为 用id表示的文档向量[(9,2)]，表示第二篇稳重中第9号单词出现了2次
                                                      #详见：http://www.52nlp.cn/%E5%A6%82%E4%BD%95%E8%AE%A1%E7%AE%97%E4%B8%A4%E4%B8%AA%E6%96%87%E6%A1%A3%E7%9A%84%E7%9B%B8%E4%BC%BC%E5%BA%A6%E4%BA%8C

class Model:
    def __init__(self, dict_filename, corpus_filename, num_topics=50):
        self.num_topics = num_topics                                                             #num_topic是指的潜在主题(topic)的数目,等于转成lsi模型以后每个文档对应的向量长度
        self.original_dict_filename = dict_filename
        self.original_corpus_filename = corpus_filename
        dict_suffix = dict_filename[dict_filename.rfind("/")+1:dict_filename.rfind(".")]         #字典  的名字
        corpus_suffix = corpus_filename[corpus_filename.rfind("/")+1:corpus_filename.rfind(".")] #语料库的名字
        self.dict_filename = 'tmp/%s.dict' % dict_suffix                                         #字典   存储路径tmp/lastfm_artists.dict
        self.corpus_filename = 'tmp/%s.mm' % corpus_suffix                                       #语料库 存储路径tmp/lastfm_tags_artists.mm
        self.lsi_filename = 'tmp/%s_%s.lsi' % (corpus_suffix, num_topics)                        #LSI模型存储路径tmp/lastfm_tags_artists_50.lsi
        self.index_filename = 'tmp/%s_%s.lsi.index' % (corpus_suffix, num_topics)                #记录索引tmp/lastfm_tags_artists_50.lsi.index
        self.doc2id_filename = "tmp/%s.doc2id.pickle" % corpus_suffix                            #临时文件tmp/lastfm_tags_artists.doc2id.pickle
        self.id2doc_filename = "tmp/%s.id2doc.pickle" % corpus_suffix                            #临时文件tmp/lastfm_tags_artists.id2doc.pickle
        self._create_directories()


    def _create_directories(self):       #生成tmp文件夹
        if not os.path.exists("tmp"):
            os.mkdir("tmp")




    def _create_docs_dict(self, docs):   #创建一个把 语料库里的文档名字 翻译成 文档标志的文档库字典
        '''
        Create the dictionaries that translate from document name to document index in the corpus, and viceversa
        '''
        self.doc2id = dict(zip(docs, range(len(docs))))              #dict(zip(['name', 'age'], ['tom', 12])) -> {'age': 12, 'name': 'tom'}
        self.id2doc = dict(zip(range(len(docs)), docs))
        pickle.dump(self.doc2id, open(self.doc2id_filename, "w"))    #pickle.dump(obj, file, [,protocol]) 将对象obj保存到文件file中去
        pickle.dump(self.id2doc, open(self.id2doc_filename, "w"))    #protocol协议版本，0：ASCII协议，1：老式的二进制协议，2：2.3版本引入的新二进制协议。 http://www.cnblogs.com/pzxbc/archive/2012/03/18/2404715.html
    
    def _load_docs_dict(self):           #加载文档库字典
        '''
        Load the dictionaries that translate from document name to document index in the corpus, and viceversa
        '''
        self.doc2id = pickle.load(open(self.doc2id_filename))        #pickle.load(file)，从file中读取一个字符串，并将它重构为原来的python对象
        self.id2doc = pickle.load(open(self.id2doc_filename))





    def _generate_dictionary(self):                      #生成字典
        print "generating dictionary..."
        documents = []
        with open(self.original_dict_filename) as f:
            documents = [[line.strip()] for line in f]   #.strip()移除字符串头尾的()内制定字符
        self.dictionary = corpora.Dictionary(documents)  #生成词典
        self.dictionary.save(self.dict_filename)         #用save函数将词典持久化
    
    def _load_dictionary(self, regenerate=False):        #加载字典对象
        '''
        Load the dictionary gensim object. If the dictionary object does not exist, or regenerate is set to True, it will generate it.
        '''
        if not os.path.exists(self.dict_filename) or regenerate is True:
            self._generate_dictionary()                                    #如没有字典对象，生成一个字典
        else:
            self.dictionary = corpora.Dictionary.load(self.dict_filename)  #如字典对象存在，加载字典






    def _generate_corpus(self):                 #生成语料库
        print "generating corpus..."
        self.corpus = []
        corpus_memory_friendly = MyCorpus(self.original_corpus_filename, self.dictionary)
        count = 0
        for vector in corpus_memory_friendly:
            self.corpus.append(vector)         #将读取到的内容添加到语料集中
            count += 1
            if count % 10000 == 0:
                print count, "vectors processed"
        self._create_docs_dict(corpus_memory_friendly.docs)            #创建文档库字典
        corpora.MmCorpus.serialize(self.corpus_filename, self.corpus)  #序列化，并store to disk, for later use，因为内存中的对象都是暂时的，无法长期驻存。http://blog.csdn.net/u012965373/article/details/69611709
        
    def _load_corpus(self, regenerate=False):   #加载语料库
        '''
        Load the corpus gensim object. If the corpus object does not exist, or regenerate is set to True, it will generate it.
        '''
        if not os.path.exists(self.corpus_filename) or regenerate is True:
            self._generate_corpus()                                    #如没有语料库对象，生成一个语料库
        else:
            self.corpus = corpora.MmCorpus(self.corpus_filename)       #如语料库对象存在，读取语料库






    def _generate_lsi_model(self, regenerate=False):     #生成潜语义标号（Latent Semantic Index）模型
        print "generating lsi models..."
        if not os.path.exists(self.lsi_filename) or regenerate is True:
            self.lsi = models.LsiModel(self.corpus, id2word=self.dictionary, num_topics=self.num_topics)  #num_topic是指的潜在主题(topic)的数目,等于转成lsi模型以后每个文档对应的向量长度
            self.lsi.save(self.lsi_filename)
            self.index = similarities.MatrixSimilarity(self.lsi[self.corpus])  #计算LSI模型语料库中文档间的相似度并记录索引
            self.index.save(self.index_filename)
        elif not os.path.exists(self.index_filename):
            self.lsi = models.LsiModel.load(self.lsi_filename)
            self.index = similarities.MatrixSimilarity(self.lsi[self.corpus])
            self.index.save(self.index_filename)
    
    def _load_lsi_model(self, regenerate=False):         #LSI和LDA的结果也可以看做该文档的文档向量
        '''
        Load the LSI and the index gensim object. If any of the two object does not exist, or regenerate is set to True, it will generate it.
        '''
        if os.path.exists(self.lsi_filename) and os.path.exists(self.index_filename) and regenerate is False:
            self.lsi = models.LsiModel.load(self.lsi_filename)
            self.index = similarities.MatrixSimilarity.load(self.index_filename)
            print self.index
        else:
            self._generate_lsi_model(regenerate)





    def load(self, regenerate=False):
        '''
        Load all the necessary objects for the model. If any object does not exist, it will generate it.
        '''
        self._load_dictionary(regenerate)
        self._load_corpus(regenerate)
        self._load_lsi_model(regenerate)
        self._load_docs_dict()





    def _get_vector(self, doc):
        vec_bow = None
        try:
            vec_bow = self.corpus[self.doc2id[doc]]
        except KeyError:
            print "Document '%s' does not exist. Have you used the proper string cleaner?" % doc
        return vec_bow
    
    def get_similars(self, doc, num_sim=20):
        '''
        Given a document (doc), this method retrieves the num_sim most similar documents in the LSI models
        '''
        vec_bow = self._get_vector(doc)
        if vec_bow is None:
            return []
        vec_lsi = self.lsi[vec_bow]      # convert the new document into the LSI space, without affecting the model
        sims = self.index[vec_lsi]       #vec_lsi保存了文档与50个topic之间的关联度
        sims = sorted(enumerate(sims), key=lambda item: -item[1])[1:num_sim+1]   #sorted使用key参数进行多条件排序，enumerate返回一个可枚举对象，lambda匿名函数
        sims = [(self.id2doc[docid], weight) for docid, weight in sims]
        return sims
    
    def get_pairwise_similarity(self, doc1, doc2):          #计算两文档的相似度
        '''
        Given two document names (doc1 and doc2), this method computes the cosine similarity between the documents' vectors in the LSI models
        '''
        vec_bow1 = self._get_vector(doc1)                   #得到文档的词向量
        vec_bow2 = self._get_vector(doc2)
        if vec_bow1 is None or vec_bow2 is None:
            return None
        vec_lsi1 = [val for idx,val in self.lsi[vec_bow1]]  #得到文档的词频向量
        vec_lsi2 = [val for idx,val in self.lsi[vec_bow2]]
        return cosine(vec_lsi1, vec_lsi2)                   #余弦相似性  http://www.ruanyifeng.com/blog/2013/03/cosine_similarity.html





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Item similarity models using Latent Semantic Indexing')
    parser.add_argument('item', help='The item name')
    parser.add_argument('-n', '--num_topics', type=int, default=100, help='Number of topics, i.e. number of factors to use from the original SVD decomposition (default=100)')
    parser.add_argument('-s', '--num_similars', type=int, default=20, help='Number of similar items to retrieve (deafult=20)')
    parser.add_argument('-c', '--cleaner', default="DefaultCleaner", help='An object that cleans the item from special chars (deafult=DefaultCleaner)')
    parser.add_argument('-r', '--regenerate_models', type=bool, default=False, help='regenerate models even if they already exist (default=False)')
    parser.add_argument('-p', '--pairwise_similarity', default=None, help='Computer pairwise similarity given a second item as argument (default=None)')
    
    args = parser.parse_args()
    
    model = Model(config.DICTIONARY_FILENAME, config.CORPUS_FILENAME, num_topics=args.num_topics)
    model.load(args.regenerate_models)
    try:
        item_cleaner = getattr(cleaner, args.cleaner)()
    except AttributeError:
        print "Cleaner '%s' does not exist. Using 'DefaultCleaner' instead" % args.cleaner
        item_cleaner = cleaner.DefaultCleaner()
    if args.pairwise_similarity is not None:
        print model.get_pairwise_similarity(item_cleaner.clean(args.item), item_cleaner.clean(args.pairwise_similarity))
    else:
        sims = model.get_similars(item_cleaner.clean(args.item), args.num_similars)
        for sim in sims:
            print sim
    