

import os
import re
import scipy
import pickle
import numpy as np
from sqlitedict import SqliteDict
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TEXT_VECTORIZER': 'text_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'starspace_embeddings.tsv',
    'STOP_WORDS': 'stopwords.pkl',
    'CHIT-CHAT_MODEL_WEIGHTS': 'chit-chat_model_weights.hdf5'
}

for key, value in RESOURCE_PATH.items():
    RESOURCE_PATH[key] = os.path.join(os.path.dirname(__file__), value)

def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()

def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    embeddings = dict()
    for line in open(embeddings_path, encoding='utf-8'):
        row = line.strip().split('\t')
        embeddings[row[0]] = np.array(row[1:], dtype=np.float32)
    embeddings_dim = embeddings[list(embeddings)[0]].shape[0]
    
    return embeddings, embeddings_dim
    

def question_to_vec(question, embeddings, dim):
    """
        question: a string
        embeddings: dict where the key is a word and a value is its' embedding
        dim: size of the representation

        result: vector representation for the question
    """
    result = np.zeros(dim)
    n = 0
    for word in question.split(' '):
        if word in embeddings:
            result += embeddings[word]
            n += 1
            
    return result / max(1, n)

def pickle_object(obj, filename, protocol=4):
    """Pickles file to the file."""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def cos_cdist(matrix, vector):
    """Computes the cosine distances between each row of matrix and vector."""
    v = vector.reshape(1, -1)
    return scipy.spatial.distance.cdist(matrix, v, 'cosine').reshape(-1)



MAX_LEN=32

ALPHABET = [' ', '!', '"', '#', '$', '%', '&', "'", ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '~', '\x92', '\x93', '\x94', '\x96', '\x97', '£', '¹', 'Ç', 'Õ', 'à', 'ä', 'ç', 'è', 'é', 'ê', 'í', 'ñ', 'ó', 'ù', 'û']

START_SYMBOL = 'START'
END_SYMBOL = 'END'
PAD_SYMBOL = 'PAD'
SPECIAL_CHARACHTERS = [PAD_SYMBOL, START_SYMBOL, END_SYMBOL]

char2id = {c:i for i, c in enumerate(SPECIAL_CHARACHTERS + ALPHABET)}
id2char = {i:c for i, c in enumerate(SPECIAL_CHARACHTERS + ALPHABET)}

LATENT_DIM = 384
EMBEDDINGS_DIM = 16
VOCAB_SIZE = len(char2id)

def text2seq(text, max_len):
    """Converts sequence of chars to sequence of indices and preserves special characters."""
    start = [char2id[START_SYMBOL]]
    chars_ids = [char2id.get(text[i], char2id[' ']) for i in range(min(max_len - 2, len(text)))]
    end = [char2id[END_SYMBOL]]
    padding = [char2id[PAD_SYMBOL]] * max(0, max_len - len(text) - 2) 

    return start + chars_ids + end + padding

def seq2text(seq, remove_special=True):
    """Converts sequence of indices to sequence of char and removes special characters."""
    text = ''.join(map(id2char.get, seq))

    if remove_special:
        for spc in SPECIAL_CHARACHTERS:
            text = text.replace(spc, ' ')

    text = re.sub(r'\s+', ' ', text).strip()
  
    return text

def GCA_response(encoder, decoder, context, max_steps=MAX_LEN):
    """Uses keras encoder and decoder models for generating response using context."""
    rnn_state = [np.zeros((1, LATENT_DIM))]

    context = np.array(text2seq(context, MAX_LEN)).reshape(1, -1)
    rnn_state = [encoder.predict([context] + rnn_state)]
    
    response_partial = np.full((1, MAX_LEN), char2id[PAD_SYMBOL])
    response_partial[0, 0] = char2id[START_SYMBOL]
    
    response = []
    for i in range(1, min(max_steps, MAX_LEN)):
        output_tokens, *rnn_state = decoder.predict([response_partial] + rnn_state)
        
        sampled_token_index = np.argmax(output_tokens[0, 0])
        if sampled_token_index == char2id[END_SYMBOL]: break
          
        response.append(sampled_token_index)
        response_partial[0, 0] = sampled_token_index
            
    text = seq2text(response, remove_special=False)

    return text
