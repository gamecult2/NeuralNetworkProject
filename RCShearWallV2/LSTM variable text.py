import numpy as np
import tensorflow as tf
from keras import backend as K, regularizers
from tensorflow.keras.layers import Layer, Input, Dense, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization, Dropout, Add
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# Define R2 metric
def r_square(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    # R, p = pearsonr(SS_res, SS_tot)
    return 1 - SS_res / (SS_tot + K.epsilon())


def generate_sequences(num_samples, max_sequence_length, num_features):
    sequences = np.zeros((num_samples, max_sequence_length, num_features))
    displacements = np.zeros(num_samples)
    for i in range(num_samples):
        seq_len = np.random.randint(40, max_sequence_length + 1)  # +1 for inclusive max
        sequences[i, :seq_len, :] = np.random.rand(seq_len, num_features)
        displacements[i] = np.mean(sequences[i])
        # displacements[i] = np.mean(sequences[i])*2
    return sequences, displacements


# Plot the data here (outside the loop)
# for i in range(num_samples):
#     plt.plot(gma[i, :, 0], color="red", label="Sequence")  # Assuming feature 0 is for plotting
#     plt.axhline(y=displacements[i], color='blue', linestyle='--', label='Average Displacement')
#     plt.xlabel("Time Step")
#     plt.ylabel("Value")
#     plt.title(f"Sequence {i + 1}")
#     plt.legend()
#     # plt.show()

# Transformer block
def transformer_block(inputs, num_heads, ff_dim):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attn_output = Dropout(0.1)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(add([inputs, attn_output]))

    ffn_output = Dense(ff_dim, activation="relu")(out1)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    ffn_output = Dropout(0.1)(ffn_output)

    return LayerNormalization(epsilon=1e-6)(add([out1, ffn_output]))


# Positional encoding function
def get_positional_encoding(sequence_length, embedding_dim):
    positions = np.arange(sequence_length)[:, np.newaxis]
    dimensions = np.arange(embedding_dim)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / np.float32(embedding_dim))
    angle_rads = positions * angle_rates

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


# Parameters
num_samples = 1000
max_sequence_length = 50
num_features = 1
embedding_dim = 64
num_heads = 8
ff_dim = 64

# Generate dummy data
gma, displacements = generate_sequences(num_samples, max_sequence_length, num_features)

# Model Definition
input_layer = Input(shape=(max_sequence_length, num_features))
# Embedding and normalization
x = Dense(embedding_dim)(input_layer)
x = RMSNorm()(x)

# Grouped-Query Attention block
attn_output = GroupedQueryAttention(embed_dim=embedding_dim, num_heads=num_heads)(x)
attn_output = Add()([x, attn_output])  # Residual connection
attn_output = RMSNorm()(attn_output)

# Feed-Forward Network block with SwiGLU
fnn_output = FNNwithSwiGLU(ff_dim=ff_dim)(attn_output)
fnn_output = Add()([attn_output, fnn_output])  # Residual connection
fnn_output = RMSNorm()(fnn_output)

# Linear layer
output_layer = Dense(1)(fnn_output)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=[r_square])

# Train the model
model.fit(gma, displacements, epochs=20, batch_size=16)

# Predict
predictions = model.predict(gma)

# Plot the data here (outside the loop)
for i in range(5):
    print(predictions[i])
    plt.axhline(y=predictions[i], color='blue', linestyle='--', label='predictions Displacement')
    plt.axhline(y=displacements[i], color='red', linestyle='--', label='original Displacement')
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title(f"Sequence {i + 1}")
    plt.legend()
    plt.show()
