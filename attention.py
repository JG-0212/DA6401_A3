import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Set seed
CELL_MAP = {
    "RNN" : keras.layers.SimpleRNN,
    "LSTM" : keras.layers.LSTM,
    "GRU" : keras.layers.GRU
}

@keras.saving.register_keras_serializable()
class BahdanauAttention(keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = keras.layers.Dense(units)
        self.W2 = keras.layers.Dense(units)
        self.V = keras.layers.Dense(1)

    def call(self, query, values):
        # query: decoder hidden state at current timestep (bs, hidden)
        # values: encoder outputs (bs, max_len, hidden)
        query_with_time_axis = tf.expand_dims(query, 1)  # (bs, 1, hidden)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(query_with_time_axis)))  # (bs, max_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)  # (bs, max_len, 1)
        context_vector = attention_weights * values  # (bs, max_len, hidden)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (bs, hidden)
        return context_vector, attention_weights

class Char2CharModelAttention:
    def __init__(self, hidden_size=256, epochs=25, batch_size=64, dropout=0,
        cell_type="LSTM", num_encoder_layers=2, num_decoder_layers=2):
        # Hyperparameters
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.cell_type = cell_type
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
            
        # Model requirements
        self.num_encoder_tokens = 0
        self.num_decoder_tokens = 0
        self.max_encoder_seq_length = 0
        self.max_decoder_seq_length = 0
        self.input_token_index = None
        self.target_token_index = None
        self.reverse_input_char_index = None
        self.reverse_target_char_index = None
        self.model = None
        self.encoder_model = None
        self.decoder_model = None

    def preprocess(self, data, train=False):
        input_texts = []
        target_texts = []
        input_characters = set('_')
        target_characters = set('_')

        for _, row in data.iterrows():
            input_text, target_text = row[1], row[0]
            if not isinstance(input_text, str) or not isinstance(target_text, str):
                continue
            target_text = "\t" + target_text + "\n"  # start and end tokens
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                input_characters.add(char)
            for char in target_text:
                target_characters.add(char)

        if train:
            input_characters = sorted(list(input_characters))
            target_characters = sorted(list(target_characters))
            self.num_encoder_tokens = len(input_characters)
            self.num_decoder_tokens = len(target_characters)
            self.max_encoder_seq_length = max(len(txt) for txt in input_texts)
            self.max_decoder_seq_length = max(len(txt) for txt in target_texts)
            self.input_token_index = {char: i for i, char in enumerate(input_characters)}
            self.target_token_index = {char: i for i, char in enumerate(target_characters)}
            self.reverse_input_char_index = {i: char for char, i in self.input_token_index.items()}
            self.reverse_target_char_index = {i: char for char, i in self.target_token_index.items()}
        else:
            # Use stored values for validation/test
            input_characters = sorted(self.input_token_index.keys())
            target_characters = sorted(self.target_token_index.keys())

        encoder_input_data = np.zeros(
            (len(input_texts), self.max_encoder_seq_length, self.num_encoder_tokens),
            dtype="float32"
        )
        decoder_input_data = np.zeros(
            (len(input_texts), self.max_decoder_seq_length, self.num_decoder_tokens),
            dtype="float32"
        )
        decoder_target_data = np.zeros(
            (len(input_texts), self.max_decoder_seq_length, self.num_decoder_tokens),
            dtype="float32"
        )

        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, self.input_token_index[char]] = 1.0
            encoder_input_data[i, t + 1 :, self.input_token_index["_"]] = 1.0
            for t, char in enumerate(target_text):
                decoder_input_data[i, t, self.target_token_index[char]] = 1.0
                if t > 0:
                    decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.0
            decoder_input_data[i, t + 1 :, self.target_token_index["_"]] = 1.0
            decoder_target_data[i, t:, self.target_token_index["_"]] = 1.0

        if train:
            print(f"Number of samples: {len(input_texts)}")
            print(f"Number of unique input tokens: {self.num_encoder_tokens}")
            print(f"Number of unique output tokens: {self.num_decoder_tokens}")
            print(f"Max sequence length for inputs: {self.max_encoder_seq_length}")
            print(f"Max sequence length for outputs: {self.max_decoder_seq_length}")

        return input_characters, target_characters, encoder_input_data, decoder_input_data, decoder_target_data

    def train(self, train_data, dev_data):
        _, _, train_encoder_input_data, train_decoder_input_data, train_decoder_target_data = self.preprocess(train_data, train=True)
        _, _, dev_encoder_input_data, dev_decoder_input_data, dev_decoder_target_data = self.preprocess(dev_data)

        set_seed(42)
        # Encoder
        encoder_inputs = keras.Input(shape=(None, self.num_encoder_tokens))
        encoder_outputs = encoder_inputs
        embedding_len = self.hidden_size
        
        for i in range(self.num_encoder_layers):
          encoder = CELL_MAP[self.cell_type](embedding_len, name=f"encoder_{i}", return_sequences=True, return_state=True, dropout=self.dropout)
          #cell state is s_t, hidden state is h_t
          encoder_outputs, state_h, state_c = encoder(encoder_outputs)
        encoder_states = [state_h, state_c]

        # Decoder inputs
        decoder_inputs = keras.Input(shape=(None, self.num_decoder_tokens))
        # Create separate attention layers (done in __init__ or train)
        self.attention_layers = [BahdanauAttention(self.hidden_size) for i in range(self.num_decoder_layers)]

        decoder_outputs = decoder_inputs
        for i in range(self.num_decoder_layers):
          decoder_cell = CELL_MAP[self.cell_type](self.hidden_size, name=f"decoder_{i}", return_sequences=True, return_state=True, dropout=self.dropout)

          # Attention layer
          all_outputs = []
          inputs = decoder_outputs
          states = encoder_states

          # Loop over decoder time steps
          for t in range(self.max_decoder_seq_length):
              decoder_input_t = keras.layers.Lambda(lambda x: x[:, t:t+1, :])(inputs)
              context_vector, _ = self.attention_layers[i](states[0], encoder_outputs)  # states[0] = hidden state h_t
              context_vector = keras.layers.Lambda(lambda x: tf.expand_dims(x, 1))(context_vector)
              decoder_combined_input = keras.layers.Concatenate(axis=-1)([decoder_input_t, context_vector])
              decoder_output, state_h, state_c = decoder_cell(decoder_combined_input, initial_state=states)
              states = [state_h, state_c]
              all_outputs.append(decoder_output)

          decoder_outputs = keras.layers.Concatenate(axis=1)(all_outputs)

        decoder_dense = keras.layers.Dense(self.num_decoder_tokens, name="dense", activation="softmax")
        decoder_outputs = decoder_dense(decoder_outputs)

        model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(
            [train_encoder_input_data, train_decoder_input_data],
            train_decoder_target_data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=([dev_encoder_input_data, dev_decoder_input_data], dev_decoder_target_data),
        )
        self.model = model
        self.predictor_setup()
        _, val_seq_acc = self.evaluate(dev_data.loc[:, 0], dev_data.loc[:, 1])
        print(f"Val_seq_acc: {val_seq_acc}")

    def predictor_setup(self):
        model = self.model
        # Encoder model
        encoder_inputs = model.input[0]  # input_1
        final_encoder_layer = self.model.get_layer(f"encoder_{self.num_encoder_layers-1}")
        encoder_outputs, state_h_enc, state_c_enc = final_encoder_layer.output  # lstm_1
        self.encoder_model = keras.Model(encoder_inputs, [encoder_outputs, state_h_enc, state_c_enc])

        decoder_inputs = model.input[1]  # input_2
        decoder_states_inputs = []

        for i in range(self.num_decoder_layers):
            decoder_state_input_h = keras.layers.Input(shape=(self.hidden_size,), name=f"decoder_state_input_h_{i}")
            decoder_state_input_c = keras.layers.Input(shape=(self.hidden_size,), name=f"decoder_state_input_c_{i}")
            encoder_outputs_input = keras.Input(shape=(None, self.hidden_size), name=f"encoder_outputs_input_{i}")
            decoder_states_inputs += [decoder_state_input_h, decoder_state_input_c, encoder_outputs_input]

        x = decoder_inputs
        decoder_states_outputs = []
        attention_weights_all = []

        for i in range(self.num_decoder_layers):
            decoder_cell = model.get_layer(f"decoder_{i}")
            state_input_h = decoder_states_inputs[3*i]
            state_input_c = decoder_states_inputs[3*i + 1]
            encoder_outputs_input = decoder_states_inputs[3*i + 2]
            context_vector, attention_weights = self.attention_layers[i](state_input_h, encoder_outputs_input)
            context_vector = keras.layers.Lambda(lambda x: tf.expand_dims(x, 1))(context_vector)
            combined_input = keras.layers.Concatenate(axis=-1)([x, context_vector])
            x, state_h, state_c = decoder_cell(combined_input, initial_state=[state_input_h, state_input_c])
            decoder_states_outputs += [state_h, state_c]
            attention_weights_all += [attention_weights]

        decoder_dense = model.get_layer("dense")
        decoder_outputs = decoder_dense(x)

        decoder_model = keras.Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs]+ decoder_states_outputs +  attention_weights_all
        )

        self.decoder_model = decoder_model


    def decode(self, words=None):
        batch_size = len(words)
        encoder_input_data = np.zeros((batch_size, self.max_encoder_seq_length, self.num_encoder_tokens), dtype="float32")

        for i, row in enumerate(words.itertuples(index=False)):
            input_text = row[0]
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, self.input_token_index[char]] = 1.0
            encoder_input_data[i, t + 1 :, self.input_token_index["_"]] = 1.0

        encoder_outputs, state_h, state_c = self.encoder_model.predict(encoder_input_data, verbose=0)

        target_seq = np.zeros((batch_size, 1, self.num_decoder_tokens))
        target_seq[:, 0, self.target_token_index["\t"]] = 1.0  # start token

        decoded_words = [""] * batch_size
        stop_conditions = [False] * batch_size

        states_value_start = [state_h, state_c, encoder_outputs]*self.num_decoder_layers
        states_value = states_value_start
        while not all(stop_conditions):
            outputs_all = self.decoder_model.predict(
                [target_seq]+states_value,
                verbose = 0
            )
            output_tokens = outputs_all[0]
            states_value = outputs_all[1:-self.num_decoder_layers]
            target_seq = np.zeros((batch_size, 1, self.num_decoder_tokens))

            for i in range(batch_size):
                sampled_token_index = np.argmax(output_tokens[i, -1, :])
                sampled_char = self.reverse_target_char_index[sampled_token_index]
                if sampled_char != "\n":
                    decoded_words[i] += sampled_char
                if sampled_char == "\n" or len(decoded_words[i]) > self.max_decoder_seq_length:
                    stop_conditions[i] = True
                target_seq[i, 0, sampled_token_index] = 1.0

            next_states = states_value
            states_value = []
            for i in range(self.num_decoder_layers):
                h = next_states[2 * i]
                c = next_states[2 * i + 1]
                e = encoder_outputs
                states_value += [h, c, e]

        return decoded_words

    def evaluate(self, native_words, romanized_words, batch_size=256):
        assert len(native_words) == len(romanized_words)
        start = 0
        total = len(native_words)
        decoded_words = []
        while start < total:
            end = min(start + batch_size, total)
            batch_df = pd.DataFrame(romanized_words.iloc[start:end])
            decoded_words += self.decode(batch_df)
            start += batch_size
        out = pd.DataFrame({"Romanized": romanized_words, "Native": native_words, "Predicted": decoded_words})
        out["Predicted"] = out["Predicted"].str.replace("_", "")
        accuracy = (out["Native"] == out["Predicted"]).mean()
        return out, accuracy
