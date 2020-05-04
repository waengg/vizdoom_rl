import tensorflow as tf
from collections import deque
from tensorflow import keras as K

# from layers import Layers 
import cv2
from random import sample
import numpy as np

class DeepQNetwork:

    def __init__(self, dims, actions, frames_per_state=4, e_prob=0.2, memsize=2000, gamma=0.95, training=False, batch_size=32):
        self.input_dims     = dims
        self.actions        = actions
        self.e_prob         = e_prob
        self.frames_per_state = frames_per_state
        # each sample is described as a 5-tuple (s_t (4 sequential states), a_t, r_t+1, s_t+1 (last 3 from s_t + 1 new), isterminal)
        self.mem            = deque(maxlen=memsize)
        self.training       = training
        self.batch_size     = batch_size
        self.gamma          = gamma

        self.model          = self.build_model()
        if self.training:
            self.compile_model()

    def add_transition(self, state):
        self.mem.append(state)

    def load_weights(self, path):
        self.model.load_weights(path)

    def preprocess(self, frame):
        array = np.array(frame)
        # assumes that the current config uses screen format GRAY8
        resized = cv2.resize(frame.copy(), self.input_dims)
        out = resized.astype('float32')
        return np.reshape(out, out.shape + (1,))

    def train(self):
        #From a batch sampled from transition memory, train the model
        if len(self.mem) < self.batch_size:
            print(f'Memory does not have enough samples to train [{len(self.mem)}/{self.batch_size}]. Skipping...')
            return
        batch = sample(self.mem, self.batch_size)
        states = [x['state'] for x in batch]
        """
            as described in Mnih et al. 2015:
            If state is terminal: y_j = reward
                            else: y_j = reward + \gamma * q(s, a, w)
        """
        y_j = [x['reward'] if x['terminal'] is True else self.future_q(x) for x in batch]
        # y_j = np.array([])


        self.model.fit(states, y_j, epochs=1, steps_per_epoch=1, batch_size=self.batch_size)
        # with tf.GradientTape() as tape:
        #     samples = self.get_samples()

            
    def future_q(self, mem_entry):
        state = self.preprocess(mem_entry['state'])
        q = self.model.predict(state)
        return mem_entry['reward'] + self.gamma * q[mem_entry['action']]

    def build_model(self):
        in_layer = K.layers.Input(self.input_dims + (self.frames_per_state,), name='state')
        x = K.layers.Conv2D(16, [8, 8], strides=(4, 4), activation='relu')(in_layer)
        x = K.layers.Conv2D(32, [4, 4], strides=(2, 2), activation='relu')(x)
        x = K.layers.Flatten()(x)
        x = K.layers.Dense(256, activation='relu')(x)
        q = K.layers.Dense(len(self.actions), name='q_values')(x)
        print('q_shape', q.shape)
        return K.Model(in_layer, q)

    @staticmethod
    def _mse_max_output(q_actual, q_pred):
        argmax = tf.cast(K.backend.argmax(q_pred), tf.int32)
        indices = tf.range(0, tf.shape(argmax)[0], dtype=tf.int32)
        i = K.backend.stack([indices, argmax], axis=1)
        max_q = tf.gather_nd(q_pred, indices)
        return K.backend.square(q_actual - max_q)

    def compile_model(self):
        optimizer = K.optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss=DeepQNetwork._mse_max_output)

    def save_weights(self):
        self

    def get_actions(self, state):
        state = state.reshape((1,) + self.input_dims + (self.frames_per_state,))
        return self.model.predict(state)[0]


q = DeepQNetwork((112, 112), [1,2,3], training=True)