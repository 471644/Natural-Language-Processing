# encoding=utf8

import os

from keras.models import Model
from keras.layers import Input, Embedding, GRU, Dense, Lambda

from utils import *

class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        question_vec = question_to_vec(question, self.word_embeddings, self.embeddings_dim)
        best_thread = cos_cdist(thread_embeddings, question_vec).argmin()

        return thread_ids[best_thread]

class DialogueManager(object):
    def __init__(self, paths):
        
        self.paths = paths
        
        #self.stopwords_set = unpickle_file(self.paths['STOP_WORDS'])

        # Intent recognition:
        self.intent_recognizer = unpickle_file(self.paths['INTENT_RECOGNIZER'])
        self.vectorizer = unpickle_file(self.paths['TEXT_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(self.paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(self.paths)
        
        # Chit-chat part:
        self.create_chitchat_bot()

    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        inp_context = Input(shape=(MAX_LEN,), dtype='int32', name='input_context')
        inp_reply = Input(shape=(MAX_LEN,), dtype='int32', name='input_utterance')

        embeddings = Embedding(output_dim=EMBEDDINGS_DIM, 
                               input_dim=VOCAB_SIZE, 
                               mask_zero=True, 
                               name='char_embeddings')

        encoder_input = embeddings(inp_context)
        decoder_input = embeddings(inp_reply)
        
        rnn_layer = GRU(LATENT_DIM, return_sequences=True, return_state=True, name='sequence_modeller')
        output_dense = Dense(VOCAB_SIZE, activation='softmax', name='output')

        encoder_outputs, *encoder_states = rnn_layer(encoder_input)
        encoder_outputs = output_dense(encoder_outputs)
        encoder_outputs = Lambda(lambda x: x, name='context_modelling')(encoder_outputs)

        decoder_outputs, *decoder_states = rnn_layer(decoder_input, initial_state=encoder_states)
        decoder_outputs = output_dense(decoder_outputs)
        decoder_outputs = Lambda(lambda x: x, name='reply_modelling')(decoder_outputs)
        
        bot_model = Model([inp_context, inp_reply], 
                          [encoder_outputs, decoder_outputs], 
                          name='generative_conversational_agent')
        bot_model.load_weights(self.paths['CHIT-CHAT_MODEL_WEIGHTS'])
        
        # Encoder
        encoder_state_input = Input(shape=(LATENT_DIM,))
        _, encoder_state = rnn_layer(encoder_input, initial_state=[encoder_state_input])

        self.encoder_model = Model([inp_context, encoder_state_input], encoder_state, name='bot_encoder')
        
        # Decoder
        decoder_state_input = Input(shape=(LATENT_DIM,))

        decoder_outputs, decoder_state = rnn_layer(decoder_input, initial_state=[decoder_state_input])
        decoder_outputs = output_dense(decoder_outputs)

        self.decoder_model = Model([inp_reply, decoder_state_input], [decoder_outputs, decoder_state], name='bot_decoder')
       
    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        
        prepared_question = text_prepare(question)
        
        features = self.vectorizer.transform([prepared_question])
        intent = self.intent_recognizer.predict(features)[0]

        # Chit-chat part:   
        if intent == 'dialogue':
            response = GCA_response(self.encoder_model, self.decoder_model, question)       
            return response
        
        # Goal-oriented part:
        else:        
            # Pass features to tag_classifier to get predictions.
            tag = self.tag_classifier.predict(features)[0]
            
            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread(question, tag)
           
            return self.ANSWER_TEMPLATE % (tag, thread_id)

