import gym
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'FlappyBird-v0' in env or 'FlappyBird-rgb-v0' in env:
        del gym.envs.registration.registry.env_specs[env]
import flappy_bird_gym

import time
import pygame
from argparse import ArgumentParser

def play_game(env=0, show_prints=False, show_gui=False, fps=100):

    if not env:
        env = flappy_bird_gym.make("FlappyBird-v0")

    nr_of_birds = 3
    obs = env.reset(nr_of_birds)

    if show_gui:
        env.render()

    prev_score = -1
    while True:
        if show_gui:
            pygame.event.pump()

        obs = env._get_observation()
        action = obs[0][1] < -0.05

        # Processing:
        obs, reward, done, scores = env.step([action, 1, 0])

        if show_prints:
            print("")
            for i in range(len(obs)):
                print("BIRD %d:\t"%i, obs[i], "\tReward:", reward[i], "\tdied:",done[i], "\tinfo:",scores[i])

        # Rendering the game:
        # (remove this two lines during training)
        if show_gui:
            env.render()
            time.sleep(1 / fps)  # FPS

        # Checking if any the player is still alive
        if all(done):
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

    play_game(show_prints=options.verbose, show_gui=options.show_gui, fps=options.fps)