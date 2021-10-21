# Building a chatbot with deep NLP

# Import Libaries

import numpy as np
import tensorflow as tf
import re
import time
from tensorflow.python.ops.array_ops import sequence_mask

from tensorflow.python.ops.data_flow_ops import initialize_all_tables

# import the dataset

lines = open('movie_lines.txt', encoding='utf-8',
             errors='ignore').read().split('\n')
conversations = open('movie_conversations.txt',
                     encoding='utf-8', errors='ignore').read().split('\n')


# map dic
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]


# Create a list of all the conversations
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(
        ' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(","))


# Getting the questions and the answers seperately
questions = []
answers = []

for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])


# Cleaning of the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"whats's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text


# Apply cleaning to all answers and questions
clean_question = []

for question in questions:
    clean_question.append(clean_text(question))

clean_answers = []

for answer in answers:
    clean_answers.append(clean_text(answer))


# Create a dictionary word to count
word2count = {}

for question in clean_question:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1


# Creating 2 DIkS with words above a threshhold

threshhold = 20
questionswords2int = {}
word_number = 0

for word, count in word2count.items():
    if count >= threshhold:
        questionswords2int[word] = word_number
        word_number += 1

answerwords2int = {}
word_number = 0

for word, count in word2count.items():
    if count >= threshhold:
        answerwords2int[word] = word_number
        word_number += 1

# Adding the last tokens to these two dictionaries

tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']

for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1
for token in tokens:
    answerwords2int[token] = len(questionswords2int) + 1


# Create the inverse dictionaries questionswords2int & answerwords2int

answersint2word = {w_i: w for w, w_i in answerwords2int.items()}

# adding end of string to every answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

# Translating clean answers and questions into unique int's
# and Replacing all the words that where filtered out by <OUT>

question2int = []

for question in clean_question:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    question2int.append(ints)


answers2int = []

for answers in clean_answers:
    ints = []
    for word in answers.split():
        if word not in answerwords2int:
            ints.append(answerwords2int['<OUT>'])
        else:
            ints.append(answerwords2int[word])
    answers2int.append(ints)


# Sorting question and answers by the length of questions

sorted_clean_question = []

sorted_clean_answers = []

for length in range(1, 25 + 1):
    for i in enumerate(question2int):
        if len(i[1]) == length:
            sorted_clean_question.append(question2int[i[0]])
            sorted_clean_answers.append(answers2int[i[0]])


# Creating placeholders for the inputs and the targets

def model_inputs():
    input = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='target')
    lr = tf.placeholder(tf.float32,  name='learning_rate')
    keep_prob = tf.placeholder(tf.float32,  name='keep_prob')
    return input, targets, lr, keep_prob

# Preprocessing the targets


def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets


# Creating the Encoder RNN layer

def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(
        lstm, input_keep_prob=keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                                    cell_bw=encoder_cell,
                                                                    sequence_length=sequence_length,
                                                                    inputs=rnn_inputs,
                                                                    dtype=tf.float32)
    return encoder_state

# Decoding the training set


def decode_training_set(encoder_state, decode_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_state = tf.zeros([batch_size, 1, decode_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(
        attention_state, attention_option='bahdanau', num_units=decode_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys, attention_values, attention_score_function, attention_construct_function, name='attn_dec_train')

    decoder_output, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
        decode_cell, training_decoder_function, decoder_embedded_input, sequence_length, scope=decoding_scope)

    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)

# decoding the test/validation set


def decode_test_set(encoder_state, decode_cell, decoder_embeddeding_matrix, sos_id, eos_id, maximus_length, num_words, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_state = tf.zeros([batch_size, 1, decode_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(
        attention_state, attention_option='bahdanau', num_units=decode_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddeding_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximus_length,
                                                                              num_words,
                                                                              name='attn_dec_inf')

    test_predictions, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decode_cell,
                                                                    test_decoder_function,
                                                                    scope=decoding_scope)
    return test_predictions


# Creating the Decoder RNN

def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(
            lstm, input_keep_prob=keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()

        def output_function(x): return tf.contrib.layers.fully_connected(x,
                                                                         num_words,
                                                                         None,
                                                                         scope=decoding_scope,
                                                                         weights_initializer=weights,
                                                                         biases_initializer=biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions


# Building the seq2seq model

def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, question_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer=tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(
        encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)

    preprocessed_targets = preprocess_targets(
        targets, questionswords2int, batch_size)

    decoder_embeddings_matrix = tf.Variable(tf.random_uniform(
        [question_num_words + 1, decoder_embedding_size], 0, 1))

    decoder_embedded_input = tf.nn.embedding_lookup(
        decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state,
                                                         question_num_words, sequence_length, rnn_size, num_layers, questionswords2int, keep_prob, batch_size)
    return training_predictions, test_predictions


# Setting the Hyperparameters
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encodding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5


tf.reset_default_graph()

session = tf.InteractiveSession()

# Loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()

# setting the sequence length
sequence_length = tf.placeholder_with_default(25, None, name='sequence_length')

# Set the input shape
input_shape = tf.shape(inputs)

# getting the training predictions and the test predictions

training_predictions, test_predictions = seq2seq_model(
    tf.reverse(inputs, [-1]),
    targets,
    keep_prob,
    batch_size,
    sequence_length,
    len(answerwords2int),
    len(questionswords2int),
    encodding_embedding_size,
    decoding_embedding_size,
    rnn_size,
    num_layers,
    questionswords2int
)
