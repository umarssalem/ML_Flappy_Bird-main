import gym
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'FlappyBird-v0' in env or 'FlappyBird-rgb-v0' in env:
        del gym.envs.registration.registry.env_specs[env]
import flappy_bird_gym

import time
import pygame
from argparse import ArgumentParser

def play_game(show_prints=False, show_gui=False, fps=100):

    env = flappy_bird_gym.make("FlappyBird-v0")
    obs = env.reset()

    if show_gui:
        env.render()

    prev_score = -1
    while True:
        if show_gui:
            pygame.event.pump()

        obs = env._get_observation()
        if obs[1] < -0.05:
            action = 1 #flap
        else:
            action = 0 #idle

        # Processing:
        obs, reward, done, info = env.step(action)

        if show_prints and prev_score != info['score']:
            prev_score = info['score']
            print(obs)
            print("\tReward:", reward, "\tdied:",done, "\tinfo:",info)

        # Rendering the game:
        # (remove this two lines during training)
        if show_gui:
            env.render()
            time.sleep(1 / fps)  # FPS

        # Checking if the player is still alive
        if done:
            break

    env.close()
    return info

if __name__=='__main__':
    parser = ArgumentParser()

    parser.add_argument("-g",
                        dest="show_gui",
                        action="store_true",
                        help="Whether the game GUI should be shown or not")

    parser.add_argument("-v", "--verbose",
                        dest="verbose",
                        action="store_true",
                        help="Print information while playing the game")

    parser.add_argument("-fps",
                        dest="fps",
                        type=int,
                        default=100,
                        help="Specify in how many FPS the game should run")

    options = parser.parse_args()

    play_game(options.verbose, options.show_gui, options.fps)