import gym
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'FlappyBird-v0' in env or 'FlappyBird-rgb-v0' in env:
        del gym.envs.registration.registry.env_specs[env]
import flappy_bird_gym

import pygame
import numpy as np
import sys
import os.path
from argparse import ArgumentParser
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import scipy.stats as st
import seaborn as sns

nr_of_birds = 5
iterations = int(500/nr_of_birds)
value = -0.054

def main(env=0, show_prints=False, show_gui=False, fps=100):
    env = flappy_bird_gym.make("FlappyBird-v0")
    bar_length = 30

    test_result = 1
    results = 0

    filename = "marco_carlo_results.npy"

    if test_result:
        start2 = datetime.now()

        results = np.zeros(iterations*nr_of_birds, dtype=int)
        for i in range(iterations):
            results[i*nr_of_birds:(i+1)*nr_of_birds] = play_game(env, show_prints=False, show_gui=False)[:]

            # Progress bar
            if i % int(iterations/100 + 1) == 0:
                percent = 100.0*i/(iterations-1)
                sys.stdout.write('\r')
                sys.stdout.write("\rExperiment progress: [{:{}}] {:>3}%".format('='*int(percent/(100.0/bar_length)),bar_length, int(percent)))
                sys.stdout.flush()

        end2 = datetime.now()
        np.save(filename, results)
        print("\nExperiment took %.2f mins\n"%((end2-start2).seconds/60))
        print("Fraction of games scored zero points: %.2f%%" %(len(results[results==0])/len(results)*100) )

    elif os.path.isfile(filename):
        with open(filename, 'rb') as f:
            results = np.load(f)
        start2 = datetime.now()
        end2 = datetime.now()

    print("average:", np.average(results))
    print("std dev:", np.std(results))
    print("variance:", np.std(results)**2)
    print("2st moment:", st.moment(results, moment=2))
    print("3st moment:", st.moment(results, moment=3))
    print("4st moment:", st.moment(results, moment=4))

    if test_result or True:
        fig, axs = plt.subplots(2,1, figsize=(10,7))
        plt.subplots_adjust(hspace=0.4)

        ax = 0
        axs[ax].plot(results, 'ob', alpha=0.1, markersize=2)
        axs[ax].set_title("After learning")
        axs[ax].set_xlabel("Iteration")
        axs[ax].set_ylabel("Points scored")
        axs[ax].set_xlim([0,len(results)])

        ax = 1
        freq, bins, patches = axs[ax].hist(results, color='red', ec="k", bins=100, weights=np.ones(len(results)) / len(results))
        axs[ax].set_ylabel("Frequency")
        axs[ax].set_xlabel("Points scored")
        axs[ax].set_title("%d games ~ took %.2f mins"%(iterations*nr_of_birds,(end2-start2).seconds/60))
        axs[ax].set_xlim(-0.25)
        axs[ax].yaxis.set_major_formatter(PercentFormatter(1))

        plt.savefig("Marco_Carlo_results.png")

        plt.figure()
        sns.histplot(data=results, kde=True, color = 'darkblue',bins=40, stat="probability")
        # sns.displot(data=results, kde=True, color = 'darkblue',bins=int(180/5))
        # sns.distplot(results, hist=True, kde=True,
        #      bins=int(180/5), norm_hist=True, color = 'darkblue',
        #      hist_kws={'edgecolor':'black'},
        #      kde_kws={'linewidth': 1})
        plt.xlabel("Points scored")
        plt.title("Baseline - %d games"%10000)
        plt.xlim(0)
        plt.savefig("Marco_Carlo_results.png")


def play_game(env=0, show_prints=False, show_gui=False, fps=100):

    if not env:
        env = flappy_bird_gym.make("FlappyBird-v0")

    obs = env.reset(nr_of_birds)

    if show_gui:
        env.render()

    while True:
        if show_gui:
            pygame.event.pump()

        actions = np.zeros(nr_of_birds)
        obs = env._get_observation()
        for i in range(nr_of_birds):
            actions[i] = (obs[i][1] < value)

        # Processing:
        obs, reward, done, scores = env.step(actions)

        if show_prints:
            print("")
            for i in range(nr_of_birds):
                print("BIRD %d:\t"%i, obs[i], "\tReward:", reward[i], "\tdied:",done[i], "\tscore:",scores[i])

        # Rendering the game:
        # (remove this two lines during training)
        if show_gui:
            env.render()
            time.sleep(1 / fps)  # FPS

        # Checking if any the player is still alive
        if all(done):
            break

    env.close()
    return scores



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

    main(show_prints=options.verbose, show_gui=options.show_gui, fps=options.fps)