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
from time import sleep
from matplotlib import pyplot as plt

USE_GPU = False

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

def build_action(n_actions, index):
    return [True if i == index else False for i in range(0, n_actions)]

def build_all_actions(n_actions):
    return [build_action(n_actions, index) for index in range(0, n_actions)]

if __name__ == "__main__":
    game = vzd.DoomGame()

    game.set_window_visible(True)

    game.set_doom_scenario_path("../scenarios/simple_corridor.wad")

    # Sets map to start (scenario .wad files can contain many maps).
    game.set_doom_map("MAP01")

    # Sets resolution. Default is 320X240
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)

    # Sets the screen buffer format. Not used here but now you can change it. Default is CRCGCB.
    game.set_screen_format(vzd.ScreenFormat.GRAY8)

    # Enables depth buffer.
    game.set_depth_buffer_enabled(True)

    # Enables labeling of in game objects labeling.
    game.set_labels_buffer_enabled(True)

    # Enables buffer with top down map of the current episode/level.
    game.set_automap_buffer_enabled(True)

    # Enables information about all objects present in the current episode/level.
    game.set_objects_info_enabled(True)

    # Enables information about all sectors (map layout).
    game.set_sectors_info_enabled(True)

    # Sets other rendering options (all of these options except crosshair are enabled (set to True) by default)
    game.set_render_hud(True)
    game.set_render_minimal_hud(False)  # If hud is enabled
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)  # Bullet holes and blood on the walls
    game.set_render_particles(False)
    game.set_render_effects_sprites(False)  # Smoke and blood
    game.set_render_messages(False)  # In-game messages
    game.set_render_corpses(False)
    game.set_render_screen_flashes(True)  # Effect upon taking damage or picking up items
    game.set_doom_skill(1)
    # Adds buttons that will be allowed.
    game.add_available_button(vzd.Button.MOVE_LEFT)
    game.add_available_button(vzd.Button.MOVE_RIGHT)
    game.add_available_button(vzd.Button.MOVE_FORWARD)
    game.add_available_button(vzd.Button.TURN_LEFT)
    game.add_available_button(vzd.Button.TURN_RIGHT)

    # Adds game variables that will be included in state.
    game.add_available_game_variable(vzd.GameVariable.AMMO2)

    # Causes episodes to finish after 200 tics (actions)
    game.set_episode_timeout(10000)

    # Makes episodes start after 10 tics (~after raising the weapon)
    game.set_episode_start_time(10)

    # Makes the window appear (turned on by default)
    #game.set_window_visible(True)

    # Turns on the sound. (turned off by default)
    game.set_sound_enabled(False)

    # Sets the livin reward (for each move) to -1
    game.set_living_reward(-1)
    game.get_game_variable
    # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
    game.set_mode(vzd.Mode.PLAYER)

    # Enables engine output to console.
    #game.set_console_enabled(True)

    # Initialize the game. Further configuration won't take any effect from now on.
    game.init()
    n_actions = game.get_available_buttons_size()
    actions = build_all_actions(n_actions)
    # find a way to create this array in a smarter way
    tf.config.experimental_run_functions_eagerly(False)
    # Run this many episodes
    episodes = 1000
    resolution = (320, 240)
    dims = (resolution[1]//3, resolution[0]//3)
    frames_per_state = 4

    dql = DeepQNetwork(dims, n_actions, training=False, dueling=True)
    # dql.load_weights('../weights/dqn_simple_dueling')
    state_buffer = deque(maxlen=4)
    sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028

    for i in range(episodes):
        while not game.is_episode_finished():
            state = game.get_state()
            frame = state.screen_buffer
            print(frame.shape)
            # plt.imshow(frame, cmap='gray')
            # plt.show()
            # sleep(0.5)
            processed_frame = dql.preprocess(frame)
            if len(state_buffer) == 0:
                [state_buffer.append(processed_frame) for _ in range(frames_per_state)]
            else:
                state_buffer.append(processed_frame)
            # fig = plt.figure()
            # for im_index, f in enumerate(state_buffer):
            #     fig.add_subplot(1, 4, im_index+1)
            #     plt.imshow(np.squeeze(f), cmap='gray')
            # plt.show()
            state_array = np.array(state_buffer)
            state_array = np.squeeze(np.rollaxis(state_array, 0, 3))
            print(state_array.shape)
            q_vals = dql.get_actions(state_array)
            print(q_vals)
            eps = 1
            if random() < eps:
                a = choice(actions)
            else:
                best_action = np.argmax(q_vals)
                a = build_action(n_actions, best_action)
            r = game.make_action(a, 4)
            if sleep_time > 0:
                sleep(sleep_time)
        state_buffer.clear()
        game.new_episode()



