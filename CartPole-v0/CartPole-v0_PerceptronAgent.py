import argparse
import logging
import sys

import numpy as np

import gym
from gym import wrappers

import theano
import lasagne

"""
Approximate the reward
"""
class RewardPerceptronAgent(object):

    def __init__(self, inputState, inputAction, outputReward):

        ## Define the network through Lasagne library based on Theano
        # self.networkDefinition(inputState, inputAction)
        self.networkDefinition(inputState, inputAction)

        ## Define the output function (WITHOUT dropout)
        prediction = lasagne.layers.get_output(self.network, deterministic=True)
        self.evaluateReward = theano.function([inputState, inputAction], prediction)

        ## Define the loss function for back-propagation (WITHOUT dropout)
        prediction = lasagne.layers.get_output(self.network, deterministic=True)
        loss = lasagne.objectives.squared_error(prediction, outputReward)
        loss = loss.mean()

        ## Define the update function
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.sgd(loss, params, learning_rate=0.2)

        ## Define the training function
        self.trainFunction = theano.function([inputState, inputAction, outputReward], loss, updates=updates)

        # Custom loss estimation for display
        prediction = lasagne.layers.get_output(self.network, deterministic=True)
        loss = lasagne.objectives.squared_error(prediction, outputReward)
        loss = loss.mean()
        self.estimateLoss = theano.function([inputState, inputAction, outputReward], loss)

    ## Define the network through Lasagne library based on Theano
    def networkDefinition(self, inputState, inputAction):
        
        """
        IMPORTANT : 

        Necessary to use "axis = -1" due tu issue #736 : https://github.com/Lasagne/Lasagne/pull/726
        Then, each time the work is done on the first dimention of an input/layer or whatever, use axis=-1
        """

        self.inputState             = inputState
        self.inputAction            = inputAction

        self.l_in_state             = lasagne.layers.InputLayer([4], input_var=inputState, name='l_in_state')
        self.l_in_action            = lasagne.layers.InputLayer([1], input_var=inputAction, name='l_in_action')

        self.l_merge_input          = lasagne.layers.ConcatLayer([self.l_in_state, self.l_in_action], axis = -1)
        # self.l_merge_input_drop     = lasagne.layers.DropoutLayer(self.l_merge_input, p=0.2)

        self.l_hidden_1             = lasagne.layers.DenseLayer(self.l_merge_input, num_units=8, num_leading_axes=-1, nonlinearity=lasagne.nonlinearities.sigmoid, b=lasagne.init.Constant(1))
        # self.l_hidden_1_drop        = lasagne.layers.DropoutLayer(self.l_hidden_1, p=0.1)
        self.l_hidden_2             = lasagne.layers.DenseLayer(self.l_hidden_1, num_units=8, num_leading_axes=-1, nonlinearity=lasagne.nonlinearities.sigmoid)
        # self.l_hidden_2_drop        = lasagne.layers.DropoutLayer(self.l_hidden_2, p=0.1)
        self.l_out                  = lasagne.layers.DenseLayer(self.l_hidden_2, num_units=1, num_leading_axes=-1, nonlinearity=lasagne.nonlinearities.sigmoid)

        self.network = self.l_out


    def act(self, observation):
        
        ## Two annotation allowed : I choose the second one, more compact but less explicite
        estimatedRewardAction0 = self.evaluateReward(observation, [0])[0]
        estimatedRewardAction1 = self.evaluateReward(observation, [1])[0]

        # print('____________________')
        # print(estimatedRewardAction0)
        # print(estimatedRewardAction1)

        if estimatedRewardAction0 < estimatedRewardAction1:
            return (0, estimatedRewardAction0)
        else:
            return (1, estimatedRewardAction1)

    def train(self, observation, action, estimatedReward, currentReward):

        # print('Current loss')
        # print(estimatedReward)
        # print(currentReward)
        # print(self.estimateLoss(observation, [action], [currentReward]))

        self.trainFunction(observation, [action], [currentReward])

def customReward(currentAngle):
    maxAllowedAngle = .26
    return 1-np.absolute(currentAngle)/maxAllowedAngle

if __name__ == '__main__':

    outdir = '/NeuralNetwork/tmp/CartPole-v0/PerceptronAgent'

    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    env.reset()

    ## Initialize the neural network
    observation = theano.tensor.vector(name='inputObservation')
    action = theano.tensor.vector(name='inputAction')
    reward = theano.tensor.vector(name='outputReward')
    agent = RewardPerceptronAgent(observation, action, reward)

    done = False

    for i in range(1000):
        print("Episode {} ".format(i))
        observation = env.reset()
        for t in range(10000):

            action, estimatedReward = agent.act(observation)
            observation, reward, done, info = env.step(action)


            cReward = customReward(observation[2])
            agent.train(observation, action, estimatedReward, cReward)

            if done:
                # print(agent.estimateLoss(observation, [action], [-1]))
                print("Episode finished after {} timesteps".format(t+1))

                # print(observation)
                # print(cReward)
                break

            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.


    # Close the env and write monitor result info to disk
    env.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    # logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    # gym.upload(outdir, api_key='sk_eYEmndURAKEkXnZmSxtEA')


