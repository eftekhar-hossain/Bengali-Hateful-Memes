import numpy as np
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Concatenate, Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten,Reshape,dot,multiply,Permute
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, Convolution1D,MaxPooling1D,Conv1D, Activation, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop,Adam,SGD,Nadam
##
from evaluation import PrintMetrics



## Multimodal Contextual Attention Mechanism
class MCA(tf.keras.Model):
    def __init__(self, units):
        super(MCA, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features shape == (batch_size, max_len, 2*lstm_units)
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = tf.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis))
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        context_vector1 = attention_weights * features
        context_vector2 = attention_weights * hidden_with_time_axis
      
        context_vector1 = tf.reduce_sum(context_vector1, axis=1)
        context_vector2 = tf.reduce_sum(context_vector2, axis=1)
        context_vector = keras.layers.concatenate([context_vector1,  context_vector2])
        return context_vector, attention_weights



# Text Guided Contextual Attention Mechanism
class TGCA(tf.keras.Model):
    def __init__(self, units):
        super(TGCA, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):

        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        score = tf.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis))
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # Text Guided
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

# Vision Guided Contextual Attention Mechanism
class VGCA(tf.keras.Model):
    def __init__(self, units):
        super(VGCA, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        score = tf.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # vision Guided
        context_vector = attention_weights * hidden_with_time_axis
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights





def MultimodalContextualFusion(train, valid, test, labels, hparams, md_hparams, class_names, method_name):
  """
  Multi-Modal Alignment
  Generate context vectors for two modalities and concatenate them with the visual and textual features.

  Arguments:
  train -- list [train_text, train_image]
  valid -- list [valid_text, valid_image]
  test -- list [test_text, test_image]
  labels -- list [train_label, valid_label, test_label]
  hparams -- list [maximum_length, img_size, vocabulary_size, embedding_dimension, no_classes]
  md_hparams -- list [model_name, batch, ep]

  Returns:
  model -- a Model() instance in Keras

  """

  train_label, valid_label, test_label = labels[0], labels[1], labels [2]
  maximum_length, img_size, vocabulary_size, embedding_dimension, no_classes =  hparams[0], hparams[1], hparams[2], hparams[3], hparams[4]
  model_name, batch, ep = md_hparams[0], md_hparams[1], md_hparams[2] 
  # visual feature extractor (ResNet50)

  base_model = ResNet50(weights='imagenet', include_top=False,input_shape=(img_size, img_size, 3))
  base_model.trainable = False
  y = base_model.output
  pool = GlobalAveragePooling2D()(y)
  dense_layer = Dense(100,activation = 'relu')(pool)
  visual_features = dense_layer

  # textual feature extractor ()
  sequence_input = Input(shape=(maximum_length,), dtype="int32")
  embedded_sequences = Embedding(vocabulary_size, embedding_dimension)(sequence_input)
  # Getting our LSTM outputs
  (lstm, forward_h, forward_c, backward_h, backward_c) = Bidirectional(LSTM(50, return_sequences=True, return_state=True), name="bi_lstm_1")(embedded_sequences)
  state_h = Concatenate()([forward_h, backward_h])

  # Attentive Fusion
  if method_name == 'mca-scf':
    context_vector, attention_weights = MCA(10)(lstm, visual_features)
    final = keras.layers.concatenate([context_vector, visual_features, state_h])
    output = Dense(no_classes, activation="softmax")(final)
  elif method_name == 'mcf':  
    context_vector, attention_weights = MCA(10)(lstm, visual_features)
    output = Dense(no_classes, activation="softmax")(context_vector)

  elif method_name == 'tgcf':
    context_vector, attention_weights = TGCA(10)(lstm, visual_features)
    final = keras.layers.concatenate([context_vector, visual_features])
    output = Dense(no_classes, activation="softmax")(final)
  
  elif method_name == 'vgcf':
    context_vector, attention_weights = VGCA(10)(lstm, visual_features)
    final = keras.layers.concatenate([context_vector, state_h])
    output = Dense(no_classes, activation="softmax")(final)
  

  model = Model(inputs=[sequence_input,base_model.input], outputs=output)

  filepath = 'Models/'+ f'{model_name}.h5'
  checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=2, save_best_only=True,
                                             save_weights_only=True, mode='max' )
  keras.backend.clear_session()
  
  #### Training 
  print(f"Start Training {method_name.upper()} ")
  model.compile(loss='sparse_categorical_crossentropy', optimizer= Adam(), metrics=['accuracy'])

  model.fit(train,train_label,
                      batch_size=batch,
                      epochs=ep,
                      validation_data = (valid, valid_label),
                      verbose = 1,
                      callbacks = [checkpoint] )
  print("Training Done")

  # Evaluation

  print(f"Start Evaluating {method_name.upper()}")
  model.load_weights('Models/'+ f'{model_name}.h5')
  pred = np.argmax(model.predict(test),axis = 1)
  #pred = (model.predict(test) > 0.5).astype(int)
  PrintMetrics(test_label, pred, class_names)
  print("Evaluation Done")



