import tensorflow as tf
from collections import deque
from tensorflow import keras as K
import datetime

# from layers import Layers
import cv2
from random import sample, uniform, randint
import numpy as np

class DeepQNetwork:

    def __init__(self, dims, n_actions, frames_per_state=4, start_eps=1.0, end_eps=0.1,
                anneal_eps=True, anneal_until=750000, memsize=90000, gamma=0.9, training=False, batch_size=32):


        self.TAG            = DeepQNetwork.__name__
        self.input_dims     = dims
        self.n_actions      = n_actions
        self.start_eps      = start_eps
        self.end_eps        = end_eps
        self.anneal_eps     = anneal_eps
        self.anneal_until   = anneal_until
        self.frames_per_state = frames_per_state
        # each sample is described as a 5-tuple (s_t (4 sequential states), a_t, r_t+1, s_t+1 (last 3 from s_t + 1 new), isterminal)
        self.mem            = deque(maxlen=memsize)
        self.training       = training
        self.batch_size     = batch_size
        self.gamma          = gamma

        self.model          = self.build_model(training=training)
        if self.training:
            self.compile_model()

    def add_transition(self, state):
        self.mem.append(state)

    def load_weights(self, path):
        self.model.load_weights(path)

    def preprocess(self, frame):
        """
            frame: a grayscale frame from ViZDoom
            out: frame resized to self.input_dims
        """
        array = np.array(frame)
        # assumes that the current config uses screen format GRAY8
        resized = cv2.resize(frame.copy(), self.input_dims[::-1])
        out = resized.astype('float32')
        return np.reshape(out, out.shape + (1,))

    def next_eps(self, frame_num):
        if frame_num > self.anneal_until:
            eps = self.end_eps
        else:
            eps = self.start_eps - ((self.start_eps - self.end_eps) / \
                self.anneal_until * frame_num)
        if frame_num % 1000 == 0:
            print(f'Using eps {eps} for frame {frame_num}.')
        return eps

    def train(self):
        #From a batch sampled from transition memory, train the model
        if len(self.mem) < self.batch_size:
            print(f'Memory does not have enough samples to train [{len(self.mem)}/{self.batch_size}]. Skipping...')
            return
        batch = sample(self.mem, self.batch_size)
        states = np.array([x['state'] for x in batch])
        actions = np.array([(i, x['action']) for i, x in enumerate(batch)])
        """
            as described in Mnih et al. 2015:
            If state is terminal: y_j = reward
                            else: y_j = reward + \gamma * q(s, a, w)
        """

        t = datetime.datetime.now()
        y_true = np.array(self._future_q(batch))
        dummy_y_true = [y_true, np.ones((len(batch),))]

        self.model.fit([states, actions, y_true], dummy_y_true, epochs=1, batch_size=self.batch_size, verbose=0)
        # with tf.GradientTape() as tape:
        #     samples = self.get_samples()

    def _future_q(self, batch):
        batch_size = len(batch)
        states = np.array([x['next_state'] for x in batch])
        q = self.model.predict([
            states,
            np.ones((batch_size, 2)),
            np.ones((batch_size, 1))
        ])[0]
        # print(q)

        argmax = np.argmax(q, axis=1)
        bellman = [self.gamma * q[i][argmax[i]] if not batch[i]['terminal'] else 0 for i in range(0, len(q))]
        return [mem_entry['reward'] + bellman[i] for i, mem_entry in enumerate(batch)]

    def build_model(self, training=True, dueling=False):
        in_layer = K.layers.Input(self.input_dims + (self.frames_per_state,), name='state')
        x = K.layers.Conv2D(16, [8, 8], strides=(4, 4), activation='relu')(in_layer)
        x = K.layers.Conv2D(32, [4, 4], strides=(2, 2), activation='relu')(x)
        # x = K.layers.Conv2D(64, [3, 3], strides=(1, 1), activation='relu')(x)
        x = K.layers.Flatten()(x)
        x = K.layers.Dense(256, activation='relu')(x)
        q = K.layers.Dense(self.n_actions, name='q_values')(x)
        if dueling:
            pass
        if training:
            transition_action = K.layers.Input((2,), name='transition_action', dtype='int32')
            y_true = K.layers.Input((1,), name="y_true", dtype='float32')
            masked_actions = K.layers.Lambda(lambda t: tf.gather_nd(t, transition_action))(q)
            loss = K.layers.Subtract()([y_true, masked_actions])
            loss = K.layers.Lambda(lambda t: K.backend.square(t), name='loss')(loss)
            return K.Model([in_layer, transition_action, y_true], [q, loss])
        else:
            return K.Model(in_layer, q)

    def compile_model(self):
        # lr_schedule = K.optimizers.schedules.ExponentialDecay(1e-2, 400000, 0.9)
        # optimizer = K.optimizers.Adam(learning_rate=lr_schedule)
        optimizer = K.optimizers.Adam()
        losses = [
            lambda y_true, y_pred: K.backend.zeros_like(y_pred),
            lambda y_true, y_pred: y_pred,
        ]
        self.model.compile(optimizer=optimizer, loss=losses)

    def save_weights(self, path):
        self.model.save_weights(f'{path}.h5')

    def load_weights(self, path):
        self.model.load_weights(f'{path}.h5')

    def get_actions(self, state):
        if state.ndim != 4:
            state = state.reshape((1,) + self.input_dims + (self.frames_per_state,))
        if self.training:
            batch_size = int(len(state) / self.batch_size)
            batch_size = batch_size if batch_size > 0 else 1
            split_batches = np.array_split(state, batch_size, axis=0)
            # add each batch processed into a unique array for later analysis
            result = []
            for batch in split_batches:
                result.extend(self.model.predict([
                    batch,
                    np.ones((len(batch), 2)),
                    np.ones((len(batch), 1)),
                ])[0])
            return np.array(result).reshape((len(state), self.n_actions))
        else:
            return self.model.predict([state])[0]
