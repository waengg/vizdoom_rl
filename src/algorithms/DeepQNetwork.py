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
                anneal_eps=True, anneal_until=60000, memsize=200000, start_learning_at=5000, gamma=0.99, training=False, dueling=False, batch_size=32):


        self.TAG            = DeepQNetwork.__name__
        self.input_dims     = dims
        self.n_actions      = n_actions
        self.start_eps      = start_eps
        self.end_eps        = end_eps
        self.anneal_eps     = anneal_eps
        self.anneal_until   = anneal_until
        self.frames_per_state = frames_per_state
        self.start_learning_at = start_learning_at
        # each sample is described as a 5-tuple (s_t (4 sequential states), a_t, r_t+1, s_t+1 (last 3 from s_t + 1 new), isterminal)
        self.mem            = deque(maxlen=memsize)
        self.training       = training
        self.batch_size     = batch_size
        self.gamma          = gamma
        self.dueling        = dueling
        self.curr_frame     = 0

        self.model          = self.build_model(training=training, dueling=dueling)
        self.model.summary()
        if self.training:
            self.compile_model()

    def build_model(self, training=True, dueling=False):
        in_layer = K.layers.Input(self.input_dims + (self.frames_per_state,), name='state')
        x = K.layers.Conv2D(32, [8, 8], strides=(4, 4))(in_layer)
        x = K.layers.Activation('relu')(x)
        x = K.layers.Conv2D(64, [4, 4], strides=(2, 2))(x)
        x = K.layers.Activation('relu')(x)
        x = K.layers.Conv2D(64, [3, 3], strides=(2, 2))(x)
        x = K.layers.Activation('relu')(x)
        x = K.layers.Flatten()(x)
        if dueling:
            # state value
            v = K.layers.Dense(256, activation='relu')(x)
            v = K.layers.Dense(1, activation='linear')(v)
            v = K.layers.Lambda(lambda s: tf.expand_dims(s[:, 0], axis=-1), output_shape=(self.n_actions),)(v)

            # action advantages
            a = K.layers.Dense(256, activation='relu')(x)
            a = K.layers.Dense(self.n_actions, activation='linear')(a)

            sub = K.layers.Subtract()([a, tf.reduce_mean(a, axis=1, keepdims=True)])
            q = K.layers.Add()([v, sub])
        else:
            x = K.layers.Dense(256, activation='relu')(x)
            q = K.layers.Dense(self.n_actions, name='q_values', activation='linear')(x)
        if training:
            mask = K.layers.Input((self.n_actions,), name='mask', dtype='float32')
            # y_true = K.layers.Input((1,), name="y_true", dtype='float32')
            # loss = K.layers.Subtract()([y_true, q])
            # loss = K.layers.Lambda(lambda t: K.backend.square(t), name='loss')(loss)
            # masked_loss = K.layers.Multiply()([loss, mask])
            # masked_output = K.layers.Multiply()([q, mask])
            masked_output = q
            # return K.Model([in_layer, mask, y_true], [q, masked_loss])
            return K.Model([in_layer, mask], [masked_output])
        else:
            return K.Model(in_layer, q)

    def _huber_loss(self, y_true, y_pred):
        clip_value = 1.
        td_error = y_true - y_pred
        condition = tf.abs(td_error) < clip_value
        squared_loss = .5 * tf.square(td_error)
        linear_loss = clip_value * (tf.abs(td_error) - .5 * clip_value)
        return tf.where(condition, squared_loss, linear_loss)

    def compile_model(self):
        # tested with:
        # 1e-1, 1e-2, 1-e3, 1e-4
        # with decay:
        # 1e-1, 1e-2, 1e-4 - 100K, 0.99, 0.5
        # 1e-2 is bad for learning. Model quickly estimates everything as the same
        #lr_schedule = K.optimizers.schedules.ExponentialDecay(1e-5, 100000, 0.02)
        # optimizer = K.optimizers.RMSprop(learning_rate=1e-4)
        optimizer = K.optimizers.RMSprop(learning_rate=2.5e-4, momentum=0.95, decay=0.95, epsilon=1e-2)
        # losses = [
        #     lambda y_true, y_pred: K.backend.zeros_like(y_pred),
        #     lambda y_true, y_pred: K.backend.zeros_like(y_pred),
        #     # lambda y_true, y_pred: tf.squeeze(y_pred),
        # ]
        loss = lambda y_true, y_pred: tf.square(y_true - y_pred)
        # self.model.compile(optimizer=optimizer, loss=losses)
        # self.model.compile(optimizer=optimizer, loss=self._huber_loss)
        self.model.compile(optimizer=optimizer, loss='mse')

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
                # model_out = self.model.predict([
                #     batch,
                #     np.zeros((len(batch), self.n_actions)),
                #     np.zeros((len(batch), 1)),
                # ])
                model_out = self.model.predict([
                    batch,
                    np.ones((len(batch), self.n_actions))
                ])
                result.extend(model_out)
            return np.array(result)
        else:
            return self.model.predict([state])

    def add_transition(self, state):
        self.mem.append(state)

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
        return np.expand_dims(out, axis=-1)

    def next_eps(self, frame_num):
        """
            Computers the next epsilon based on a linear decay

            frame_num: current frame number
            eps: epsilon to use on the current action step
        """
        if frame_num < self.start_learning_at:
            return self.start_eps
        if frame_num > self.anneal_until:
            eps = self.end_eps
        else:
            eps = self.start_eps - ((self.start_eps - self.end_eps) / \
                self.anneal_until * (frame_num - self.start_learning_at))
        if frame_num % 1000 == 0:
            print(f'Using eps {eps} for frame {frame_num}.')
        return eps

    def train(self, second_model=None, memory=None):

        #From a batch sampled from transition memory, train the model
        if memory is not None:
            indexes, batch, is_weights = memory.sample(self.batch_size)
            is_weights = np.squeeze(is_weights)
            states = np.array([exp[0][0] for exp in batch])
            actions = np.array([exp[0][1] for exp in batch])
        else:
            if len(self.mem) < self.start_learning_at:
                return None
            batch = sample(self.mem, self.batch_size)
            states = np.array([x['state'] for x in batch])
            actions = np.array([x['action'] for x in batch])

        t = datetime.datetime.now()
        mask = np.zeros((len(batch), self.n_actions))
        if memory is not None:
            discounted_reward = self._future_q(batch, second_model, prioritized=True)
        else:
            discounted_reward = np.array(self._future_q(batch, second_model))
        q = self.model.predict([
            states,
            np.ones((len(batch), self.n_actions))])
        # y_true = np.zeros((len(batch), self.n_actions))
        y_true = q.copy()
        # print("discounted reward:", discounted_reward)
        td_error = np.zeros((self.batch_size,))
        for i, action in enumerate(actions):
            mask[i][action] = 1.
            y_true[i][action] = discounted_reward[i]
            if memory is not None:
                td_error[i] = np.abs(discounted_reward[i] - q[i][action])
        if memory is not None:
            memory.batch_update(indexes, td_error)
            
        # y_true = np.array(self._future_q(batch, second_model))
        # dummy_y_true = [y_true, np.ones((len(batch),))]

        # return self.model.fit([states, mask, y_true], dummy_y_true, epochs=1, batch_size=self.batch_size, verbose=0)
        # return self.model.fit([states, mask], y_true, epochs=1, batch_size=self.batch_size, verbose=0)
        if memory is not None:
            # loss = self.model.train_on_batch([states, mask], y_true, sample_weight=is_weights)
            loss = self.model.train_on_batch([states, mask], y_true)
            # q = self.model.predict([
            #     states,
            #     np.ones((len(batch), self.n_actions))])
            return loss
        else:
            return self.model.train_on_batch([states, mask], y_true)
            # return self.model.train_on_batch([states, np.ones((len(self.batch_size), self.n_actions))], y_true)

    def _future_q(self, batch, second_model=None, prioritized=False):
        self.curr_frame += 1
        batch_size = len(batch)
        if prioritized:
            rewards = np.array([exp[0][2] for exp in batch])
            states = np.array([exp[0][3] for exp in batch])
            is_terminal = np.array([exp[0][4] for exp in batch])
        else:
            rewards = np.array([x['reward'] for x in batch])
            states = np.array([x['next_state'] for x in batch])
            is_terminal = np.array([x['terminal'] for x in batch])
            
        # q = self.model.predict([
        #     states,
        #     np.ones((batch_size, self.n_actions)),
        #     np.ones((batch_size, 1))
        # ])[0]
        q = self.model.predict([
            states,
            np.ones((batch_size, self.n_actions))])

        if second_model:
            # q_hat = second_model.model.predict([
            #     states,
            #     np.ones((batch_size, self.n_actions)),
            #     np.ones((batch_size, 1))
            # ])[0]
            q_hat = second_model.model.predict([
                states,
                np.ones((batch_size, self.n_actions))])
            #     argmax = np.argmax(q_hat, axis=1)
        # else:
        argmax = np.argmax(q, axis=1)
        """
            as described in Mnih et al. 2015:
            If state is terminal: y_j = reward
                            else: y_j = reward + \gamma * q(s_t+1, a, w) (DQN)
                            or    y_j = reward + \gamma * q(s_t+1, argmax_a q'(s_t+1, a, w'), w) (Double DQN)
        """
        # transition_action_q = []
        # for i in range(len(q)):
        #     transition_action_q.append(q[i][argmax[i]])
        # transition_action_q = np.array(transition_action_q)
        bellman = [self.gamma * q_hat[i][argmax[i]] if not is_terminal[i] else 0. for i in range(len(q))]
        # print('rewards:')
        # for mem_entry in batch:
        #     print(mem_entry['reward'])
        discounted_reward = np.array([rewards[i] + bellman[i] for i in range(len(batch))])
        return discounted_reward


    def save_weights(self, path):
        self.model.save_weights(f'{path}.h5')

    def load_weights(self, path):
        self.model.load_weights(f'{path}.h5')
