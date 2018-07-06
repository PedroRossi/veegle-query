from string import punctuation
from copy import deepcopy
import numpy as np
import nltk

class QueryProcessor:

    def __init__(self, documents, index_dictionary, tfidf_enabled = False):
        self.documents = documents
        self.index_dictionary = index_dictionary
        self.attributes_list = []
        for key in self.index_dictionary:
            self.attributes_list.append(key)
        self.tfidf_enabled = tfidf_enabled
        self.__process_documents()

    def __process_documents(self):
        self.original_documents = deepcopy(self.documents)
        documents = self.documents
        for doc in documents:
            for attr in self.attributes_list:
                doc[attr] = QueryProcessor.clean_string(doc[attr])
        self.documents = documents

    @staticmethod
    def clean_string(st):
        stemmer = nltk.stem.RSLPStemmer()
        punctuation_list = list(punctuation)
        dictionary = punctuation_list + list(nltk.corpus.stopwords.words('portuguese'))
        st = ''.join([i for i in st if not i.isdigit() and i not in punctuation_list])
        st = st.split()
        st = [stemmer.stem(t) for t in st if t not in dictionary]
        return st

    def simple_search(self, query):
        query = QueryProcessor.clean_string(query)
        documents_to_get = []
        for attr in self.attributes_list:
            for q in query:
                comp = attr + '.' + q
                if comp in self.index_dictionary[attr]:
                    documents_to_get += self.index_dictionary[attr][comp]
        documents_to_get = np.unique([doc[1] for doc in documents_to_get])
        a = []
        results = []
        for q in query:
            a.append(1)
        for i in documents_to_get:
            aux = []
            for attr in self.attributes_list:
                aux += self.documents[i][attr]
            b = []
            for q in query:
                # TODO
                if self.tfidf_enabled:
                    pass
                else:
                    b.append(1 if q in aux else 0)
            ret = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
            results.append([ret, self.original_documents[i]])
        results = sorted(results, key=lambda x: x[0], reverse=True)
        results = [i[1] for i in results]
        return results

    def advanced_search(self, query):
        for attr in self.attributes_list:
            if attr in query:
                query[attr] = self.clean_string(query[attr])
        documents_to_get = []
        for key in query:
            for q in query[key]:
                comp = key + '.' + q
                if comp in self.index_dictionary[key]:
                    documents_to_get += self.index_dictionary[key][comp]
        documents_to_get = np.unique([doc[1] for doc in documents_to_get])
        a = []
        results = []
        for key in query:
            for q in query[key]:
                a.append(1)
        for i in documents_to_get:
            b = []
            for key in query:
                for q in query[key]:
                    # TODO
                    if self.tfidf_enabled:
                        pass
                    else:
                        b.append(1 if q in self.documents[i][key] else 0)
            ret = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
            results.append([ret, self.original_documents[i]])
        results = sorted(results, key=lambda x: x[0], reverse=True)
        results = [i[1] for i in results]
        return results