import argparse
import logging
import sys

import numpy as np

import gym
from gym import wrappers


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()



if __name__ == '__main__':

    outdir = '/NeuralNetwork/tmp/CartPole-v0/RandomAgent'

    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    env.reset()


    agent = RandomAgent(env.action_space)

    episode_count = 1
    reward = 0
    done = False


    wholeAction = []
    wholeObservation = []

    for i in range(episode_count):
        observation = env.reset()
        for t in range(1000):

            action = agent.act(observation, reward, done)
            observation, reward, done, info = env.step(action)

            wholeAction += [action]
            wholeObservation += [observation[2]]

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.


        print(wholeAction)
        print(wholeObservation)

        thefile = open('/NeuralNetwork/test.txt', 'w')
        thefile.write("%s\n" % wholeAction)
        thefile.write("\n")
        thefile.write("%s\n" % wholeObservation)


    # Close the env and write monitor result info to disk
    env.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    # gym.upload(outdir, api_key='sk_eYEmndURAKEkXnZmSxtEA')
