#!/usr/bin/env python3

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
from shutil import rmtree
from math import sqrt

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

# tf.compat.v1.disable_eager_execution()
from algorithms.DeepQNetwork import DeepQNetwork
import numpy as np
from vizdoom import vizdoom as vzd

# print(gpus)
import numpy as np

def build_action(n_actions, index):
    return [True if i == index else False for i in range(n_actions)]

def build_all_actions(n_actions):
    return [build_action(n_actions, index) for index in range(n_actions)]

def setup_tensorboard(path):
    current_time = datetime.datetime.now().strftime('%d%m%Y-%H%M%S')
    
    train_log_dir = f'{path}/{current_time}/train'
    print(train_log_dir)
    os.makedirs(train_log_dir, exist_ok=True)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    return train_summary_writer

def write_tensorboard_data(writer, episode, episode_reward, episode_loss, accumulated_reward):
    with writer.as_default():
        tf.summary.scalar('Episode Reward', episode_reward, step=episode)
        tf.summary.scalar('Average Episode Loss', episode_loss, step=episode)
        tf.summary.scalar('Accumulated Reward', accumulated_reward, step=episode)
    writer.flush()

def write_avg_q(writer, avg_q, step):
    with writer.as_default():
        tf.summary.scalar('Average Q', avg_q, step=step)
    writer.flush()

# both state acquisitions seem to be working as intended
def build_memory_state(state, action, reward, new_state, is_terminal):
    state_array = np.array(state)
    state_array = np.squeeze(np.rollaxis(state_array, 0, 3))
    new_state_array = np.array(new_state)
    new_state_array = np.squeeze(np.rollaxis(new_state_array, 0, 3))
    return {
        'state': state_array,
        'action': action,
        'reward': reward,
        'next_state': new_state_array,
        'terminal': is_terminal
    }

# method seems to be handling state acquisition well
def dry_run(game, n_states, actions, available_maps):
    visited_states = []
    state_buffer = deque(maxlen=4)
    game.new_episode()
    for i in range(n_states):
        #TODO: refactor state collection and preprocessing into a single function
        state = game.get_state()
        frame = state.screen_buffer
        processed_frame = dql.preprocess(frame)
        if len(state_buffer) == 0:
            [state_buffer.append(processed_frame) for _ in range(4)]
        else:
            state_buffer.append(processed_frame)
        state_buffer_array = np.array(state_buffer)

        state_buffer_array = np.squeeze(np.rollaxis(state_buffer_array, 0, 3))
        # fig = plt.figure()
        # for im_index in range(4):
        #     fig.add_subplot(1, 4, im_index+1)
        #     plt.imshow(np.squeeze(state_buffer_array[:,:, im_index]), cmap='gray')
        # plt.show()
        visited_states.append(np.squeeze(state_buffer_array))
        game.make_action(choice(actions), 3)
        if game.is_episode_finished():
            state_buffer.clear()
            # game.close()
            # setup_game(game, choice(available_maps))
            game.new_episode()
    return np.array(visited_states)

def eval_average_q(states, network):
    q_vals = network.get_actions(states)
    print(q_vals[200:230])
    max_values = np.max(q_vals, axis=1)
    return np.mean(max_values)

def fill_memory(memory, game, dqn, actions):
    print("Prioritized Memory: Filling prioritized memory with random samples...")
    game.new_episode()
    state_buffer = deque(maxlen=4)
    for i in range(memory.capacity):
        if i % 5000 == 0:
            print(f"Prioritized Memory: Frame {i}")
        state = game.get_state()
        frame = state.screen_buffer
        processed_frame = dqn.preprocess(frame)
        if len(state_buffer) == 0:
            [state_buffer.append(processed_frame) for _ in range(4)]
        else:
            state_buffer.append(processed_frame)
        action = randint(0, actions-1)
        action_array = build_action(actions, action)
        reward = game.make_action(action_array, 3)
        if reward < -1:
            reward = -1.
        elif reward > 1.:
            reward = 1.
        next_state_buffer = state_buffer.copy()
        is_terminal = game.is_episode_finished()
        np_state = create_np_state(state_buffer)
        if not is_terminal:
            next_frame = game.get_state().screen_buffer
            processed_next_frame = dql.preprocess(frame)
            next_state_buffer.append(processed_next_frame)
        else:
            state_buffer.clear()
            # game.close()
            # setup_game(game, choice(available_maps))
            game.new_episode()
        np_next_state = create_np_state(next_state_buffer)
        experience = np_state, action, reward, np_next_state, is_terminal
        memory.store(experience)
    print("done filling memory")

def create_np_state(state_buffer):
    state_array = np.array(state_buffer)
    state_array = np.squeeze(np.rollaxis(state_array, 0, 3))
    return state_array

def limit_gpu_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def select_random_map(available_maps):
    chosen_map = choice(available_maps)

def setup_game(game, wad):

    print(f'Setting up map {wad["name"]}')

    game.load_config(f"../scenarios/configs/{wad['cfg']}")
    game.set_doom_scenario_path(f"../scenarios/{wad['name']}")
    game.set_doom_map(wad['map'])
    game.init()

def create_parser():
    pass

def minimum_distance_traveled(curr_pos, past_pos, min_difference):
    curr_x, curr_y = curr_pos
    past_x, past_y = past_pos

    distance = sqrt((curr_x - past_x)**2 + (curr_y - past_y)**2)

    if distance > min_difference: 
        return True
    else:
        return False

if __name__ == "__main__":
    train_name = 'nature_arch_dtc_3_frameskip_0.1'
    # Create DoomGame instance. It will run the game and communicate with you.

    # #TODO: remove every ViZDoom configuration code and create a cfg file containing them
    available_maps = [
        # {'name': 'fork_corridor.wad', 'map': 'MAP01'},
        # {'name': 'simple_corridor.wad', 'map': 'MAP01', 'cfg': 'training.cfg'},
        # {'name': 'simple_corridor_distance.wad', 'map': 'MAP01', 'cfg': 'training.cfg'},
        # {'name': 'my_way_home.wad', 'map': 'MAP01', 'cfg': 'my_way_home.cfg'},
        {'name': 'defend_the_center-0.1.wad', 'map': 'MAP01', 'cfg': 'defend_the_center.cfg'},
        # {'name': 'deadly_corridor.wad', 'map': 'MAP01', 'cfg': 'deadly_corridor.cfg'},
        # {'name': 'basic.wad', 'map': 'map01', 'cfg': 'basic.cfg'},
        # {'name': 't_corridor.wad', 'map': 'MAP01'},
        # {'name': 'doom1_converted.wad', 'map': 'E1M1', 'cfg': 'training_fullmap.cfg'},
        # {'name': 'doom1_e1m1_door1.wad', 'map': 'E1M1', 'cfg': 'training_fullmap.cfg'},
        # {'name': 'doom_entire_converted.wad', 'map': 'E1M1', 'cfg': 'training_monsters.cfg'},
    ]
    game = vzd.DoomGame()
    setup_game(game, choice(available_maps))
    n_actions = game.get_available_buttons_size()

    actions = build_all_actions(n_actions)

    tf.config.experimental_run_functions_eagerly(False)
    
    # HYPERPARAMETERS
    episodes = 1000000
    update_steps = 1000
    resolution = (320, 240)
    dims = (resolution[1]//4, resolution[0]//4)
    frames_per_state = 4
    update_after = 4
    update_step = 0
    account_time_reward = False
    account_dist_reward = False
    min_dist_traveled   = False
    negative_reward_on_use = False
    minimum_steps_to_move = 15
    min_difference = 64.
    last_avg_q = -np.inf

    accumulated_reward = 0.

    dql = DeepQNetwork(dims, n_actions, frames_per_state=frames_per_state, training=True, dueling=True)
    dql_target = DeepQNetwork(dims, n_actions, frames_per_state=frames_per_state, training=True, dueling=True)
        # dql.load_weights('../weights/standard_arch_e1m1_breadcrumbs_r_for_moving')
        # dql_target.load_weights('../weights/standard_arch_e1m1_breadcrumbs_r_for_moving')

    # priority_memory = Memory(capacity=10000)
    # fill_memory(priority_memory, game, dql, n_actions)

    tb_writer = setup_tensorboard(f'../logs/{train_name}')
    state_buffer = deque(maxlen=frames_per_state)

    try:
        eval_states = dry_run(game, 1000, actions, available_maps)
        print("Dry run finished.")
        # setup_game(game, choice(available_maps))
        frame_number = 0
        t = datetime.datetime.now()
        for i in range(episodes):
            curr_step = 0
            tic = 0
            loss = 0.
            total_reward = 0.
            train_steps = 0
            didnt_move = 0
            isterminal = False
            past_x = -np.inf
            past_y = -np.inf
            avg_q = -np.inf
            game.new_episode()
            timeout = game.get_episode_timeout() // frames_per_state
            initial_distance = vzd.doom_fixed_to_double(game.get_game_variable(vzd.GameVariable.USER2))
            start = time.time()
            while not isterminal:
                frame_number += 1
                tic += 1.
                if frame_number % (update_after * 1500) == 0:
                    print(f'Frame {frame_number}: Updating model weights.')

                    update_step += 1
                    dql_target.model.set_weights(dql.model.get_weights())
                    # weights = dql.model.get_weights()
                    # target_weights = dql_target.model.get_weights()
                    # new_weights = []
                    # for w, tw in zip(weights, target_weights):
                    #     new_weights.append(0.01 * tw + (1 - 0.01) * w)
                    # dql.model.set_weights(new_weights)
                    print(f'Collecting Average Q for weights of episode {i}...')
                    avg_q = eval_average_q(eval_states, dql)
                    print(f'Episode {i}: Average Q: {avg_q}')
                t = datetime.datetime.now()
                state = game.get_state()
                frame = state.screen_buffer
                processed_frame = dql_target.preprocess(frame)
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
                    best_action = np.argmax(q_vals, axis=1)[0]

                a = build_action(n_actions, best_action)
                r = game.make_action(a, 3)
                isterminal = game.is_episode_finished()

                if negative_reward_on_use:
                    if best_action == n_actions-1:
                        if not isterminal:
                            r -= 0.5

                if account_time_reward:
                    tic_reward = tic / timeout
                    r -= tic_reward

                if account_dist_reward:
                    dist = vzd.doom_fixed_to_double(game.get_game_variable(vzd.GameVariable.USER1))
                    dist = dist / initial_distance
                    r -= dist

                if min_dist_traveled:
                    curr_x = vzd.doom_fixed_to_double(game.get_game_variable(vzd.GameVariable.USER1))
                    curr_y = vzd.doom_fixed_to_double(game.get_game_variable(vzd.GameVariable.USER2))
                    if past_x == -np.inf or past_y == -np.inf:
                        past_x = curr_x
                        past_y = curr_y
                    else:
                        if minimum_distance_traveled((curr_x, curr_y), (past_x, past_y), min_difference):
                            r += 0.25
                            didnt_move = 0
                            past_x = -np.inf
                            past_y = -np.inf
                        else:
                            didnt_move += 1
                        if didnt_move == 5:
                            past_x = -np.inf
                            past_y = -np.inf
                            didnt_move = 0
                    # else:
                    #     if not minimum_distance_traveled((curr_x, curr_y), (past_x, past_y), min_difference):
                    #         didnt_move += 1
                    #         if didnt_move > minimum_steps_to_move:
                    #             print(f'Frame {frame_number}: Agent did not move for {minimum_steps_to_move} frames. Ending episode.')
                    #             r -= 1.
                    #             isterminal = True
                    #             game.new_episode()
                    #     else:
                    #         didnt_move = 0
                    #         past_x = -np.inf
                    #         past_y = -np.inf
                if r < -1:
                    r = -1.
                    # print("Agent found some positive reward:", r, ". Action performed:", a)
                elif r > 1.:
                    r = 1.
                total_reward += r
                if isterminal:
                    print("Terminal state. Current tic:", tic)
                    new_state_buffer = state_buffer.copy()
                else:
                    new_state = game.get_state()
                    new_frame = new_state.screen_buffer
                    processed_new_frame = dql.preprocess(new_frame)
                    new_state_buffer = state_buffer.copy()
                    new_state_buffer.append(processed_new_frame)
                np_state = create_np_state(state_buffer)
                np_next_state = create_np_state(new_state_buffer)
                # experience = np_state, best_action, r, np_next_state, isterminal
                # priority_memory.store(experience)
                memory_state = build_memory_state(state_buffer, best_action, r, new_state_buffer, isterminal)
                dql.add_transition(memory_state)
                if frame_number % update_after == 0:
                    train_steps += 1
                    # history = dql_target.train(dql, priority_memory)
                    history = dql.train(dql)
                    if history is not None:
                        loss += history
            episode_loss = loss / train_steps if train_steps > 0 else 0.
            diff = time.time() - start
            state_buffer.clear()

            # METRICS
            accumulated_reward += total_reward
            print(f'End of episode {i}. Episode reward: {total_reward}. Episode loss: {episode_loss}. Time to finish episode: {str(diff)}')
            write_tensorboard_data(tb_writer, i, total_reward, episode_loss, accumulated_reward)
            if avg_q != -np.inf:
                write_avg_q(tb_writer, avg_q, update_step)
                if avg_q > last_avg_q:
                    print(f'Average Q {avg_q} greater than last average Q {last_avg_q}.')
                    last_avg_q = avg_q
                else:
                    print(f'Average Q {avg_q} lower than last average Q {last_avg_q}.')

            dql_target.save_weights(f'../weights/{train_name}')


    except Exception as e:
        traceback.print_exc()
        print(e)
        game.close()
        exit(1)
    game.close()
