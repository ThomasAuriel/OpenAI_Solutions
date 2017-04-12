clear

rm -rf /NeuralNetwork/tmp/CartPole-v0/PIDAgent

# https://github.com/openai/gym#rendering-on-a-server
xvfb-run -s "-screen 0 1400x900x24" bash

python CartPole-v0_PIDAgent.py
