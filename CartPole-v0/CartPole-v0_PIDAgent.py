import argparse
import logging
import sys

import numpy as np

import gym
from gym import wrappers


class PIDAgent(object):

    # previous_error = 0
    # integral = 0 
    # start:
    #   error = setpoint - measured_value
    #   integral = integral + error*dt
    #   derivative = (error - previous_error)/dt
    #   output = Kp*error + Ki*integral + Kd*derivative
    #   previous_error = error
    #   wait(dt)
    #   goto start

    def __init__(self, action_space):

        self.error = 0
        self.previous_error = 0
        self.derivative = 0
        self.integral = 0

        self.Kp = -10
        self.Ki = -1
        self.Kd = -100

    def act(self, observation, reward, done):

        self.previous_error = self.error
        self.error = 0-observation[2]
        self.derivative = self.error - self.previous_error
        self.integral = self.integral + self.error

        self.output = self.Kp*self.error + self.Ki*self.integral + self.Kd*self.derivative

        if self.output < 0:
            return 0
        else:
            return 1



if __name__ == '__main__':

    outdir = '/NeuralNetwork/tmp/CartPole-v0/PIDAgent'

    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    env.reset()

    agent = PIDAgent(env.action_space)

    episode_count = 1
    reward = 0
    done = False


    for i in range(episode_count):
        observation = env.reset()
        for t in range(1000):

            action = agent.act(observation, reward, done)
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.


    # Close the env and write monitor result info to disk
    env.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    # gym.upload(outdir, api_key='sk_eYEmndURAKEkXnZmSxtEA')
