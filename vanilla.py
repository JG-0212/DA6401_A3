import os
import random
import keras
import numpy as np
import tensorflow as tf
import pandas as pd

CELL_MAP = {
    "RNN" : keras.layers.SimpleRNN,
    "LSTM" : keras.layers.LSTM,
    "GRU" : keras.layers.GRU
}

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

class Char2CharModel:
  def __init__(self, hidden_size=32, epochs=10, batch_size=256, dropout=0,
        cell_type="LSTM", num_encoder_layers=3, num_decoder_layers=3):

    #hyperparameters
    self.hidden_size = hidden_size
    self.epochs = epochs
    self.batch_size = batch_size
    self.dropout = dropout
    self.cell_type = cell_type
    self.num_encoder_layers = num_encoder_layers
    self.num_decoder_layers = num_decoder_layers

    #model reqs
    self.num_encoder_tokens = 0
    self.num_decoder_tokens = 0
    self.max_encoder_seq_length = 0
    self.max_decoder_seq_length = 0
    self.input_token_index = None
    self.target_token_index = None
    self.reverse_input_char_index = None
    self.reverse_target_char_index = None
    self.model = None
    self.encoder_model=None
    self.decoder_model=None

  def preprocess(self, data, train=False):

    input_texts = []
    target_texts = []
    #Adding "_" as a padding character
    input_characters = set('_')
    target_characters = set('_')
    for index, row in data.iterrows():
        input_text, target_text, attesters = row[1], row[0], row[2]
        if isinstance(target_text, str) != True or isinstance(input_text, str) != True:
          continue
        target_text = "\t" + target_text + "\n"
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    if train == True:
        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))
        num_encoder_tokens = len(input_characters)
        num_decoder_tokens = len(target_characters)
        max_encoder_seq_length = max([len(txt) for txt in input_texts])
        max_decoder_seq_length = max([len(txt) for txt in target_texts])

        input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
        target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

        reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
        reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())
    else:
        num_encoder_tokens = self.num_encoder_tokens
        num_decoder_tokens = self.num_decoder_tokens
        max_encoder_seq_length = self.max_encoder_seq_length
        max_decoder_seq_length = self.max_decoder_seq_length
        input_token_index = self.input_token_index
        target_token_index = self.target_token_index

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype="float32",
    )
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype="float32",
    )
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype="float32",
    )

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        encoder_input_data[i, t + 1 :, input_token_index["_"]] = 1.0
        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        decoder_input_data[i, t + 1 :, target_token_index["_"]] = 1.0
        decoder_target_data[i, t:, target_token_index["_"]] = 1.0

    if train==True:
      self.num_encoder_tokens = num_encoder_tokens
      self.num_decoder_tokens = num_decoder_tokens
      self.max_encoder_seq_length = max_encoder_seq_length
      self.max_decoder_seq_length = max_decoder_seq_length
      self.input_token_index = input_token_index
      self.target_token_index = target_token_index
      self.reverse_input_char_index = reverse_input_char_index
      self.reverse_target_char_index = reverse_target_char_index

      print("Number of samples:", len(input_texts))
      print("Number of unique input tokens:", num_encoder_tokens)
      print("Number of unique output tokens:", num_decoder_tokens)
      print("Max sequence length for inputs:", max_encoder_seq_length)
      print("Max sequence length for outputs:", max_decoder_seq_length)

    return input_characters, target_characters, encoder_input_data, decoder_input_data, decoder_target_data

  def train(self, train_data, dev_data):

    set_seed(42)
    (train_input_characters, train_target_characters, train_encoder_input_data,
     train_decoder_input_data, train_decoder_target_data) = self.preprocess(train_data, train=True)

    (_, _, dev_encoder_input_data,
     dev_decoder_input_data, dev_decoder_target_data) = self.preprocess(dev_data)

    num_encoder_tokens = len(train_input_characters)
    num_decoder_tokens = len(train_target_characters)


    encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))  #(bs, max_seq_len, num_encoder_tokens)
    encoder_outputs = encoder_inputs
    for i in range(self.num_encoder_layers):
      embedding_len = self.hidden_size
      encoder = CELL_MAP[self.cell_type](embedding_len, return_sequences=True, return_state=True, dropout=self.dropout)
      #cell state is s_t, hidden state is h_t
      if self.cell_type == "LSTM":
          encoder_outputs, state_h, state_c = encoder(encoder_outputs)
          # (bs, max_encoder_seq_length, embedding_len), (bs, embedding_len), (bs, embedding_len)
      else:
          encoder_outputs, state_h = encoder(encoder_outputs)
          # (bs, max_encoder_seq_length, embedding_len), (bs, embedding_len)

    if self.cell_type == "LSTM":
        encoder_states = [state_h, state_c]
    else:
        encoder_states = [state_h]
    decoder_inputs = keras.Input(shape=(None, num_decoder_tokens)) #(bs, max_decoder_seq_len, num_encoder_tokens)
    decoder_outputs = decoder_inputs
    for i in range(self.num_decoder_layers):
      decoder_cell = CELL_MAP[self.cell_type](self.hidden_size, return_sequences=True, return_state=True, dropout=self.dropout)
      if self.cell_type == "LSTM":
          decoder_outputs, _, _ = decoder_cell(decoder_outputs, initial_state=encoder_states)
      else:
          if i==0:
              decoder_outputs, _ = decoder_cell(decoder_outputs, initial_state=tuple(encoder_states))
          else:
              decoder_outputs, _ = decoder_cell(decoder_outputs)

    decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        [train_encoder_input_data, train_decoder_input_data],
        train_decoder_target_data,
        batch_size=self.batch_size,
        epochs=self.epochs,
        validation_data=([dev_encoder_input_data, dev_decoder_input_data],
        dev_decoder_target_data)
    )

    self.model = model
    self.predictor_setup()
    _,val_seq_acc = self.evaluate(dev_data.loc[:,0], dev_data.loc[:,1])
    print(f"Val_seq_acc:{val_seq_acc}")

  def predictor_setup(self):

    #model.layers input1, encoder blocks, input2 decoder blocks
    model = self.model
    hidden_size = self.hidden_size

    encoder_inputs = model.input[0]  # input_1
    final_encoder_layer = self.num_encoder_layers+1
    if self.cell_type == "LSTM":
        encoder_outputs, state_h_enc, state_c_enc = model.layers[final_encoder_layer].output  # lstm_1
        encoder_states = [state_h_enc, state_c_enc]
    else:
        encoder_outputs, state_h_enc = model.layers[final_encoder_layer].output  # lstm_1
        encoder_states = [state_h_enc]

    encoder_model = keras.Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]  # input_2
    # Create inputs for each decoder layer's states
    decoder_states_inputs = []

    for i in range(self.num_decoder_layers):
        decoder_state_input_h = keras.layers.Input(shape=(hidden_size,), name=f"decoder_state_input_h_{i}")
        if self.cell_type == "LSTM":
            decoder_state_input_c = keras.layers.Input(shape=(hidden_size,), name=f"decoder_state_input_c_{i}")
            decoder_states_inputs += [decoder_state_input_h, decoder_state_input_c]
        else:
            decoder_states_inputs += [decoder_state_input_h]

    x = decoder_inputs
    decoder_states_outputs = []

    for i in range(self.num_decoder_layers):
        decoder_cell = model.layers[1 + self.num_encoder_layers + i + 1]  # adjust index
        if self.cell_type == "LSTM":
            state_input_h = decoder_states_inputs[2*i]
            state_input_c = decoder_states_inputs[2*i + 1]
            x, state_h, state_c = decoder_cell(x, initial_state=[state_input_h, state_input_c])
            decoder_states_outputs += [state_h, state_c]
        else:
            state_input_h = decoder_states_inputs[i]
            if i==0:
                x, state_h = decoder_cell(x, initial_state=[state_input_h])
            else:
                x, state_h = decoder_cell(x)
            decoder_states_outputs += [state_h]
    decoder_dense = model.layers[2 + self.num_encoder_layers + self.num_decoder_layers]
    decoder_outputs = decoder_dense(x)

    decoder_model = keras.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states_outputs
    )

    self.encoder_model = encoder_model
    self.decoder_model = decoder_model

  def decode(self, words=None):
    batch_size = len(words)
    encoder_input_data = np.zeros(
        (batch_size, self.max_encoder_seq_length, self.num_encoder_tokens),
        dtype="float32",
    )
    i = 0
    for _, row in words.iterrows():
        input_text = row.iloc[0]
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, self.input_token_index[char]] = 1.0
        encoder_input_data[i, t + 1 :, self.input_token_index["_"]] = 1.0
        i += 1
    if self.cell_type == "LSTM":
        hs, cs = self.encoder_model.predict(encoder_input_data, verbose=0)
        states_value = [hs, cs]*self.num_decoder_layers
    else:
        hs = self.encoder_model.predict(encoder_input_data, verbose=0)
        states_value = [np.copy(hs) for _ in range(self.num_decoder_layers)]
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((batch_size, 1, self.num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[:, 0, self.target_token_index["\t"]] = 1.0
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_conditions = [False]*batch_size
    decoded_words = [""]*batch_size
    while not all(stop_conditions):
        results = self.decoder_model.predict(
            [target_seq] + states_value, verbose=0
        )
        output_tokens_all = results[0]
        target_seq = np.zeros((batch_size, 1, self.num_decoder_tokens))
        for i,_ in enumerate(output_tokens_all):
            # Sample a token
            sampled_token_index = np.argmax(output_tokens_all[i, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            if sampled_char!="\n":
                decoded_words[i] += sampled_char
            # Exit condition: either hit max length
            # or find stop character.
            if sampled_char == "\n" or len(decoded_words[i]) > self.max_decoder_seq_length:
                stop_conditions[i] = True
            # Update the target sequence (of length 1).
            target_seq[i, 0, sampled_token_index] = 1.0
        # Update states
        states_value = results[1:]
    return decoded_words

  def evaluate(self, native_words, romanized_words, batch_size=256):
    assert len(native_words) == len(romanized_words)
    start = 0
    total = len(native_words)
    decoded_words = []
    while start<total:
        end = min(start+batch_size, total)
        decoded_words += self.decode(pd.DataFrame(romanized_words.iloc[start:end]))
        start += batch_size
    out = pd.DataFrame({"Romanized": romanized_words, "Native":native_words, "Predicted": decoded_words})
    out.loc[:,"Predicted"] = out.loc[:,"Predicted"].str.replace("_","")
    accuracy = (out.loc[:,"Native"] == out.loc[:,"Predicted"]).sum()/len(out)
    # print(f"Accuracy on test set: {accuracy}")
    return out, accuracy