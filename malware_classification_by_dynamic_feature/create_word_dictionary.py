from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from keras_preprocessing.text import Tokenizer
#tf-idf
with open("dynamic_feature_train.csv.pkl", "rb") as f:
    labels = pickle.load(f)#train 类别
    files = pickle.load(f)#files 文件api

vectorizer = TfidfVectorizer(ngram_range=(1, 5), min_df=3, max_df=0.9, )  # tf-idf特征抽取ngram_range=(1,5)，如果词的df超过某一阈值则被词表过滤
train_features = vectorizer.fit_transform(files) #将文本中的词语转换为词频矩阵 ,先拟合数据再标准化

tfidftransformer_path = 'tfidf_transformer.pkl'
with open(tfidftransformer_path, 'wb') as fw:
    pickle.dump(vectorizer, fw)

#deep learning


with open("dynamic_feature_test.csv.pkl", "rb") as f:
    test_labels = pickle.load(f)
    outfiles = pickle.load(f)

tokenizer = Tokenizer(num_words=None,
                      filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
                      split=' ',
                      char_level=False,
                      oov_token=None)#Tokenizer是一个用于向量化文本,或将文本转换为序列

tokenizer.fit_on_texts(files)
tokenizer.fit_on_texts(outfiles)

pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))