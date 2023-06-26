import jieba
import re

def preprocess(dataset="./dataset/dataset_cn.txt"):
    with open("./dataset/stopwords.dic", 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f]

    with open(dataset, 'r', encoding='utf-8') as f:
        documents = [document.strip() for document in f]

    word2id = {}
    id2word = {}
    docs = []
    cur_id = 0

    for document in documents:
        cur_doc = []
        seg_list = jieba.cut(document)
        for word in seg_list:
            word = word.lower().strip()
            if len(word) > 1 and not re.search('[0-9]', word) and word not in stopwords:
                if word in word2id:
                    id=word2id[word]
                else:
                    id=cur_id
                    word2id[word] = cur_id
                    id2word[cur_id] = word
                    cur_id += 1
                flag=True
                for tu in cur_doc:
                    if tu[0] == id:
                        tu[1] += 1
                        flag=False
                        break
                if flag:
                    cur_doc.append([id,1])
        docs.append(cur_doc)
    return docs, word2id, id2word