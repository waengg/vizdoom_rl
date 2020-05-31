#!/usr/bin/env python3

#####################################################################
# This script presents how to use the most basic features of the environment.
# It configures the engine, and makes the agent perform random actions.
# It also gets current state and reward earned with the action.
# <episodes> number of episodes are played.
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
#
# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

from __future__ import print_function

from random import choice
import traceback
import datetime
from collections import deque
import argparse
import tensorflow as tf
print(f'Running TensorFlow v{tf.__version__}')
import os
from random import random, randint
import time
from time import sleep
from matplotlib import pyplot as plt

USE_GPU = True
DEVICES = None
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if USE_GPU:
    DEVICES = tf.config.experimental.list_physical_devices('GPU')
    for gpu in DEVICES:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    DEVICES = tf.config.list_physical_devices('CPU')
    tf.config.experimental.set_visible_devices(devices=DEVICES, device_type='CPU')
    tf.config.experimental.set_visible_devices(devices=[], device_type='GPU')

tf.compat.v1.disable_eager_execution()
from algorithms.DeepQNetwork import DeepQNetwork
import numpy as np
from vizdoom import vizdoom as vzd

# print(gpus)
import numpy as np

def build_action(n_actions, index):
    return [True if i == index else False for i in range(0, n_actions)]

def build_all_actions(n_actions):
    return [build_action(n_actions, index) for index in range(0, n_actions)]

def build_memory_state(state, action, reward, new_state, is_terminal):
    state_array = np.array(state)
    # state_array = state_array.reshape(sa_shape[1:3] + (4,))
    state_array = np.squeeze(np.rollaxis(state_array, 0, 3))
    new_state_array = np.array(new_state)
    # new_state_array = new_state_array.reshape(nsa_shape[1:3] + (4,))
    new_state_array = np.squeeze(np.rollaxis(new_state_array, 0, 3))
    return {
        'state': state_array,
        'action': action,
        'reward': reward,
        'next_state': new_state_array,
        'terminal': is_terminal
    }

def dry_run(game, n_states, actions, available_maps):
    visited_states = []
    state_buffer = deque(maxlen=4)
    game.new_episode()
    for i in range(n_states):
        print(i)
        #TODO: refactor state collection and preprocessing into a single function
        state = game.get_state()
        frame = state.screen_buffer
        processed_frame = dql.preprocess(frame)
        if len(state_buffer) == 0:
            [state_buffer.append(processed_frame) for _ in range(4)]
        else:
            state_buffer.append(processed_frame)
        state_buffer_array = np.array(state_buffer)

        state_buffer_array = np.rollaxis(np.expand_dims(state_buffer_array, axis=-1), 0, 3)
        # test = state_buffer_array[0,:,:]
        # test = np.reshape(state_buffer_array, shape[1:3] + (4,))
        # plt.imshow(np.squeeze(test), cmap='gray')
        # plt.show()
        # sleep(0.5)
        visited_states.append(np.squeeze(state_buffer_array))
        #TODO: plot visited stated, just to ensure that they actually make sense
        game.make_action(choice(actions))
        if game.is_episode_finished():
            state_buffer.clear()
            game.close()
            setup_game(game, choice(available_maps))
            game.new_episode()
    return np.array(visited_states)

def eval_average_q(states, network):
    q_vals = network.get_actions(states)
    print(q_vals[500:520])
    argmax = np.argmax(q_vals, axis=1)
    max_values = np.array([q_vals[i][argmax[i]] for i in range(len(argmax))])
    return np.mean(max_values)

def limit_gpu_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def select_random_map(available_maps):
    chosen_map = choice(available_maps)

def setup_game(game, wad):

    print(f'Setting up map {wad["name"]}')

    # Now it's time for configuration!
    # load_config could be used to load configuration instead of doing it here with code.
    # If load_config is used in-code configuration will also work - most recent changes will add to previous ones.
    game.load_config(f"../scenarios/configs/{wad['cfg']}")

    # Sets path to additional resources wad file which is basically your scenario wad.
    # If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
    game.set_doom_scenario_path(f"../scenarios/{wad['name']}")

    # Sets map to start (scenario .wad files can contain many maps).
    game.set_doom_map(wad['map'])

    # Initialize the game. Further configuration won't take any effect from now on.
    game.init()

if __name__ == "__main__":
    # Create DoomGame instance. It will run the game and communicate with you.

    # #TODO: remove every ViZDoom configuration code and create a cfg file containing them
    available_maps = [
        # {'name': 'fork_corridor.wad', 'map': 'MAP01'},
        # {'name': 'simple_corridor.wad', 'map': 'MAP01', 'cfg': 'training.cfg'},
        {'name': 'deadly_corridor.wad', 'map': 'MAP01', 'cfg': 'deadly_corridor.cfg'},
        # {'name': 'basic.wad', 'map': 'map01', 'cfg': 'basic.cfg'},
        # {'name': 't_corridor.wad', 'map': 'MAP01'},
    ]
    game = vzd.DoomGame()
    setup_game(game, choice(available_maps))
    n_actions = game.get_available_buttons_size()

    actions = build_all_actions(n_actions)

    # find a way to create this array in a smarter way

    tf.config.experimental_run_functions_eagerly(False)
    # Run this many episodes
    episodes = 10000
    resolution = (320, 240)
    dims = (resolution[1]//4, resolution[0]//4)
    frames_per_state = 4

    dql = DeepQNetwork(dims, n_actions, training=True, dueling=True)
    state_buffer = deque(maxlen=4)

    #TODO: simplify game loop: collect state -> perform action -> collect next state -> train
    #TODO: check each and every line of this code. something MUST be off, it's impossible dude

    try:
        eval_states = dry_run(game, 10000, actions, available_maps)
        setup_game(game, choice(available_maps))
        frame_number = 0
        t = datetime.datetime.now()
        for i in range(episodes):
            print(f'Collecting Average Q for weights of episode {i}...')
            print(f'Episode {i}: Average Q: {eval_average_q(eval_states, dql)}')
            game.new_episode()
            tic = 0
            cumulative_reward = 0.
            start = time.time()
            while not game.is_episode_finished():
                t = datetime.datetime.now()
                state = game.get_state()
                frame = state.screen_buffer
                processed_frame = dql.preprocess(frame)
                if len(state_buffer) == 0:
                    [state_buffer.append(processed_frame) for _ in range(frames_per_state)]
                else:
                    state_buffer.append(processed_frame)
                rand = random()
                epsilon = dql.next_eps(frame_number)
                if rand <= epsilon:
                    best_action = randint(0, n_actions-1)
                else:
                    state_array = np.array(state_buffer)
                    state_array = np.expand_dims(np.squeeze(np.rollaxis(state_array, 0, 3)), axis=0)
                    q_vals = dql.get_actions(state_array)
                    best_action = np.argmax(q_vals)

                frame_number += 1

                before_action = datetime.datetime.now()
                a = build_action(n_actions, best_action)

                r = game.make_action(a, 4)
                cumulative_reward += r
                if r > 90:
                    print(f'I guess the bot did find the end: {r}')
                diff = datetime.datetime.now() - before_action
                # print(f'Time passed to perform one action on vizdoom: {str(diff)}')
                isterminal = game.is_episode_finished()
                if isterminal:
                    new_state_buffer = state_buffer.copy()
                else:
                    new_state = game.get_state()
                    new_frame = new_state.screen_buffer
                    processed_new_frame = dql.preprocess(new_frame)
                    new_state_buffer = state_buffer.copy()
                    new_state_buffer.append(processed_new_frame)
                memory_state = build_memory_state(state_buffer, best_action, r, new_state_buffer, isterminal)
                dql.add_transition(memory_state)
                dql.train()
                # print(f'Time to complete one training cycle: {time.time() - start}')
            diff = time.time() - start
            state_buffer.clear()
            game.close()
            setup_game(game, choice(available_maps))
                # print(f'Time passed to conclude a training cycle: {str(diff)}')

            print(f'End of episode {i}. Episode reward: {cumulative_reward}. Time to finish episode: {str(diff)}')
            dql.save_weights('../weights/dqn_simple_dueling')

    except Exception as e:
        traceback.print_exc()
        print(e)
        game.close()
        exit(1)
    # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
    game.close()
