import numpy
import collections
import re
import preprocessor
preprocessor.set_options(
    preprocessor.OPT.URL,
    preprocessor.OPT.MENTION,
    preprocessor.OPT.RESERVED)

def load_embeddings(
    embeddings_file, 
    word_index, 
    max_words, 
    embedding_dim, 
    encoding='utf8', 
    vector_dtype='float32'):

    embedding_matrix = numpy.empty((max_words, embedding_dim))

    with open(embeddings_file, 'r', encoding=encoding) as f_in:
        for line in f_in:
            word, vec = line.split(' ', 1)
            if word not in word_index:
                continue
            i = word_index[word]
            if i >= max_words:
                continue
            embedding_vector = numpy.asarray(vec.split(' '), dtype=vector_dtype)
            if embedding_vector.shape[0] == embedding_dim:
                embedding_matrix[i] = embedding_vector

    return embedding_matrix

# taken from https://github.com/cbaziotis/keras-utilities/blob/master/kutilities/helpers/data_preparation.py
def get_class_weights(y, smooth_factor=0):
    '''
    Returns the normalized weights for each class based on the frequencies of the samples
    :param smooth_factor: factor that smooths extremely uneven weights
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    '''
    counter = collections.Counter(y)

    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p

    majority = max(counter.values())

    return {cls: float(majority / count) for cls, count in counter.items()}

def text_preprocessor(doc):
    # separate hyperlinks from adjacent text, e.g. goodbypic.twitter.com -> goodbye pic.twitter.com
    doc = re.sub(r'(\w*)(https?|pic\.)', r'\1 \2', doc)
    # uniformize twitter-specific tokens
    doc = preprocessor.tokenize(doc)
    # extract text from *, e.g. *nope* -> nope
    doc = re.sub(r'\*(.*?)\*', r'\1', doc)
    # replace & symbol
    doc = re.sub(r'&', r' and ', doc)
    # lower-casing
    doc = doc.lower()
    # uniformize some corpus specific errors
    doc = re.sub(r'xan ', r'xanax ', doc)
    doc = re.sub(r'rogain', r'rogaine ', doc)
    doc = re.sub(r'adderal|aderall', r'adderall', doc)
    # normalizer multiple occurences of vowels/consonants
    doc = re.sub(r'(\w)\1\1+', r'\1', doc)
    # remove reddit symbol /r/
    doc = re.sub(r'/r/', r'', doc)
    # remove text between {}
    doc = re.sub(r'\{(.*?)\}', r'', doc)
    # uniformize emojis and numbers
    preprocessor.set_options(
        preprocessor.OPT.EMOJI,
        preprocessor.OPT.NUMBER)
    # split NUMBER and EMOJI when adjacent to text
    doc = preprocessor.tokenize(doc)
    doc = re.sub(r'(w*)(EMOJI|NUMBER)', r'\1 \2', doc)
    doc = re.sub(r'(EMOJI|NUMBER)(w*)', r'\1 \2', doc)
    # remove non-alphanumeric characters
    doc = re.sub('[^A-Za-z ]+', '', doc)
    # remove very long words >=15 and short words <2
    doc = ' '.join([item for item in doc.split() if 1 < len(item) < 18 ])
    # lower-casing
    doc = doc.lower()
    # remove multiple sequential occurences of the same token
    doc = re.sub(r'(\w+) \1 \1+', r'\1', doc)
    return doc