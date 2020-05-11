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
from time import sleep
from collections import deque
import argparse
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
limit_gpu_usage()
# tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
# gpus = tf.config.experimental.list_physical_devices('GPU')
# # gpu = list(filter(lambda g: g.name.split(':')[-1] == '0', gpus))[0]
# for gpu in gpus:    
#     tf.config.experimental.set_memory_growth(gpu, True)
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
    sa_shape = state_array.shape
    state_array = state_array.reshape(sa_shape[1:3] + (4,))
    new_state_array = np.array(new_state)
    nsa_shape = state_array.shape
    new_state_array = new_state_array.reshape(nsa_shape[1:3] + (4,))
    return {
        'state': state_array,
        'action': action,
        'reward': reward,
        'new_state': new_state_array,
        'terminal': is_terminal
    }

def limit_gpu_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def create_game():
    pass

if __name__ == "__main__":
#    gpus = tf.config.experimental.list_physical_devices('GPU')
#    limit_gpu_usage('0')
# Create DoomGame instance. It will run the game and communicate with you.
    game = vzd.DoomGame()
    game.set_window_visible(False)
    # Now it's time for configuration!
    # load_config could be used to load configuration instead of doing it here with code.
    # If load_config is used in-code configuration will also work - most recent changes will add to previous ones.
    # game.load_config("../../scenarios/basic.cfg")

    # Sets path to additional resources wad file which is basically your scenario wad.
    # If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
    game.set_doom_scenario_path("../scenarios/basic.wad")

    # Sets map to start (scenario .wad files can contain many maps).
    game.set_doom_map("map01")

    # Sets resolution. Default is 320X240
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

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
    game.set_render_hud(False)
    game.set_render_minimal_hud(False)  # If hud is enabled
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)  # Bullet holes and blood on the walls
    game.set_render_particles(False)
    game.set_render_effects_sprites(False)  # Smoke and blood
    game.set_render_messages(False)  # In-game messages
    game.set_render_corpses(False)
    game.set_render_screen_flashes(True)  # Effect upon taking damage or picking up items

    # Adds buttons that will be allowed.
    game.add_available_button(vzd.Button.MOVE_LEFT)
    game.add_available_button(vzd.Button.MOVE_RIGHT)
    game.add_available_button(vzd.Button.ATTACK)

    # Adds game variables that will be included in state.
    game.add_available_game_variable(vzd.GameVariable.AMMO2)

    # Causes episodes to finish after 200 tics (actions)
    game.set_episode_timeout(200)

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

    # Define some actions. Each list entry corresponds to declared buttons:
    # MOVE_LEFT, MOVE_RIGHT, ATTACK
    # game.get_available_buttons_size() can be used to check the number of available buttons.
    # 5 more combinations are naturally possible but only 3 are included for transparency when watching.
    actions = [[True, False, False], [False, True, False], [False, False, True]]
    # find a way to create this array in a smarter way

    # Run this many episodes
    episodes = 10
    resolution = (640, 480)
    dims = (resolution[1]//4, resolution[0]//4)
    frames_per_state = 4

    dql = DeepQNetwork(dims, actions, training=True)
    state_buffer = deque(maxlen=4)

    try:
        for i in range(episodes):
            game.new_episode()
            while not game.is_episode_finished():
                state = game.get_state()
                frame = state.screen_buffer
                processed_frame = dql.preprocess(frame)
                if len(state_buffer) == 0:
                    state_buffer.extend([processed_frame for _ in range(0, frames_per_state)])
                else:
                    state_buffer.append(processed_frame)
                # processed_state = dql.preprocess(state_buffer)
                actions = dql.get_actions(np.array(state_buffer))
                best_action = np.argmax(actions)
                #TODO: add action to make_action: requires building the action
                r = game.make_action()
                new_state = game.get_state()
                new_frame = new_state.screen_buffer
                processed_new_frame = dql.preprocess(new_frame)
                new_state_buffer = state_buffer.copy()
                new_state_buffer.append(processed_new_frame)
                isterminal = game.is_episode_finished()
                memory_state = build_memory_state(state_buffer, best_action, r, new_state_buffer, isterminal)
                dql.add_transition(memory_state)
                dql.train()


                
        
        # Sets time that will pause the engine after each action (in seconds)
        # Without this everything would go too fast for you to keep track of what's happening.
        sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028

        for i in range(episodes):
            print("Episode #" + str(i + 1))

            # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
            game.new_episode()

            while not game.is_episode_finished():

                # Gets the state
                state = game.get_state()

                # Which consists of:
                n = state.number
                variables = state.game_variables
                screen_buf = state.screen_buffer
                depth_buf = state.depth_buffer
                labels_buf = state.labels_buffer
                automap_buf = state.automap_buffer
                labels = state.labels
                objects = state.objects
                sectors = state.sectors

                # Games variables can be also accessed via:
                #game.get_game_variable(GameVariable.AMMO2)

                # Makes a random action and get remember reward.
                r = game.make_action(choice(actions))

                # Makes a "prolonged" action and skip frames:
                # skiprate = 4
                # r = game.make_action(choice(actions), skiprate)

                # The same could be achieved with:
                # game.set_action(choice(actions))
                # game.advance_action(skiprate)
                # r = game.get_last_reward()

                # Prints state's game variables and reward.
                print("State #" + str(n))
                print("Game variables:", vars)
                print("Reward:", r)
                print("=====================")

                if sleep_time > 0:
                    sleep(sleep_time)

            # Check how the episode went.
            print("Episode finished.")
            print("Total reward:", game.get_total_reward())
            print("************************")
    except Exception as e:
        print(e)
        game.close()
    # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
    game.close()
