import tensorflow as tf
from collections import deque
from tensorflow import keras as K
import datetime
from time import sleep

# from layers import Layers
import cv2
from random import sample, uniform, randint
import numpy as np

class DeepQNetwork:

    def __init__(self, dims, n_actions, frames_per_state=4, start_eps=1.0, end_eps=0.1,
                anneal_eps=True, anneal_until=120000, memsize=100000, gamma=0.99, training=False, dueling=False, batch_size=32):


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
        self.dueling        = dueling

        self.model          = self.build_model(training=training, dueling=dueling)
        self.model.summary()
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
        resized = cv2.resize(array.copy(), self.input_dims[::-1])
        out = resized / 255.
        # out = array.astype('float32')
        return np.reshape(out, out.shape + (1,))

    def next_eps(self, frame_num):
        """
            Computers the next epsilon based on a linear decay

            frame_num: current frame number
            eps: epsilon to use on the current action step
        """
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
        if len(self.mem) < 1000:
            # print(f'Memory does not have enough samples to train [{len(self.mem)}/{self.batch_size}]. Skipping...')
            return None
        batch = sample(self.mem, self.batch_size)
        states = np.array([x['state'] for x in batch])
        actions = np.array([(i, x['action']) for i, x in enumerate(batch)])

        t = datetime.datetime.now()
        mask = np.zeros((len(batch), self.n_actions))
        for i, action in actions:
            mask[i][action] = 1.
        y_true = np.array(self._future_q(batch))
        dummy_y_true = [y_true, np.ones((len(batch),))]

        return self.model.fit([states, mask, y_true], dummy_y_true, epochs=1, batch_size=self.batch_size, verbose=0)

    def _future_q(self, batch):
        batch_size = len(batch)
        states = np.array([x['next_state'] for x in batch])
        q = self.model.predict([
            states,
            np.ones((batch_size, self.n_actions)),
            np.ones((batch_size, 1))
        ])[0]
        # print("Q values por next states")
        # print(q)
        # sleep(0.5)

        """
            as described in Mnih et al. 2015:
            If state is terminal: y_j = reward
                            else: y_j = reward + \gamma * q(s, a, w)
        """
        argmax = np.argmax(q, axis=1)
        # discounted_reward = np.zeros(len(batch), self.n_actions)

        bellman = [self.gamma * q[i][argmax[i]] if not batch[i]['terminal'] else 0. for i in range(len(q))]
        # print("Discounted reward")
        # print(bellman)
        # sleep(0.5)
        return [mem_entry['reward'] + bellman[i] for i, mem_entry in enumerate(batch)]

    def build_model(self, training=True, dueling=False):
        in_layer = K.layers.Input(self.input_dims + (self.frames_per_state,), name='state')
        x = K.layers.Conv2D(16, [4, 4], strides=(4, 4))(in_layer)
        x = K.layers.Activation('relu')(x)
        x = K.layers.BatchNormalization()(x)
        x = K.layers.Conv2D(32, [4, 4], strides=(2, 2))(x)
        x = K.layers.Activation('relu')(x)
        x = K.layers.BatchNormalization()(x)
#        x = K.layers.Conv2D(32, [4, 4], strides=(2, 2))(x)
#        x = K.layers.Activation('relu')(x)
#        x = K.layers.BatchNormalization()(x)
        # x = K.layers.Conv2D(96, [2, 2], strides=(2, 2), activation='relu')(x)
        # x = K.layers.Activation('relu')(x)
        # x = K.layers.BatchNormalization()(x)
        # x = K.layers.GlobalAveragePooling2D()(x)
        x = K.layers.Flatten()(x)
        # x = K.layers.Dense(256, activation='relu')(x)
        if dueling:
            v = K.layers.Dense(256, activation='relu')(x)
            v = K.layers.BatchNormalization()(v)
            v = K.layers.Dense(1, activation='linear')(v)
            a = K.layers.Dense(128, activation='relu')(x)
            a = K.layers.BatchNormalization()(a)
            a = K.layers.Dense(self.n_actions, activation='linear')(a)
            sub = K.layers.Subtract()([a, tf.reduce_mean(a, axis=1)])
            q = K.layers.Add()([v, sub])
        else:
            x = K.layers.Dense(512, activation='relu')(x)
            x = K.layers.BatchNormalization()(x)
            q = K.layers.Dense(self.n_actions, name='q_values')(x)
        if training:
            mask = K.layers.Input((self.n_actions,), name='mask', dtype='float32')
            y_true = K.layers.Input((1,), name="y_true", dtype='float32')
            loss = K.layers.Subtract()([y_true, q])
            loss = K.layers.Lambda(lambda t: K.backend.square(t), name='loss')(loss)
            masked_loss = K.layers.Multiply()([loss, mask])
            return K.Model([in_layer, mask, y_true], [q, masked_loss])
        else:
            return K.Model(in_layer, q)

    def compile_model(self):
        # tested with:
        # 1e-1, 1e-2, 1-e3, 1e-4
        # with decay:
        # 1e-1, 1e-2, 1e-4 - 100K, 0.99, 0.5
        lr_schedule = K.optimizers.schedules.ExponentialDecay(1e-4, 100000, 0.05)
        optimizer = K.optimizers.RMSprop(learning_rate=lr_schedule)
        # optimizer = K.optimizers.Adam()
        losses = [
            lambda y_true, y_pred: K.backend.zeros_like(y_pred),
            lambda y_true, y_pred: tf.squeeze(y_pred),
        ]
        self.model.compile(optimizer=optimizer, loss=losses)

    def save_weights(self, path):
        self.model.save_weights(f'{path}.h5')

    def load_weights(self, path):
        self.model.load_weights(f'{path}.h5')

    def get_actions(self, state):
        if state.ndim != 4:
            state = np.expand_dims(state, axis=0)
        if self.training:
            batch_size = int(len(state) / self.batch_size)
            batch_size = batch_size if batch_size > 0 else 1
            split_batches = np.array_split(state, batch_size, axis=0)
            # add each batch processed into a unique array for later analysis
            result = []
            for batch in split_batches:
                model_out = self.model.predict([
                    batch,
                    np.zeros((len(batch), self.n_actions)),
                    np.zeros((len(batch), 1)),
                ])
                result.extend(model_out[0])
            return np.array(result).reshape((len(state), self.n_actions))
        else:
            return self.model.predict([state])[0]
