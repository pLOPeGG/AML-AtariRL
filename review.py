#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import gym
import random
import time
from os import path
import argparse

# Loads a saved Network
def load(sess, dir, meta):
    saver = tf.train.import_meta_graph(path.join(dir, meta))
    saver.restore(sess, tf.train.latest_checkpoint(dir))
    graph = tf.get_default_graph()
    output = graph.get_tensor_by_name('Squeeze:0')
    return output

# Choses an action with 5% randomness
def choose_action(session, model, state):
    no_action = int(model.shape[1])
    if random.random() < 0.05:
        return random.randint(0, no_action-1)
    else:
        return np.argmax(session.run(model, feed_dict={'Placeholder:0': np.reshape(state, (-1, 128, 1))}))

# Plays one game
def play_game(session, model, rand=False):
    env = gym.make('Seaquest-ram-v0')

    no_action = int(model.shape[1])
    state = env.reset()
    sum_reward = 0
    buffer_memory = []
    while(True):
        env.render()
        time.sleep(0.1)
        
        if rand:
            action = random.randint(0, no_action - 1)
        else:
            action = choose_action(session, model, state)

        framereward = 0
        done = False
        next_state = None
        for i in range(4):
            next_state, reward, done, info = env.step(action)
            framereward += reward
            if(done):
                break
        if(done):
            next_state = None

        # reward store was here
        buffer_memory.append((state, action, framereward, next_state))

        state = next_state
        sum_reward += framereward
        if(done):
            break
    return sum_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replay Seaquest game.')
    parser.add_argument('--agent', '-a', default='random', choices=['random', '1', '2', '3'], help='Chose the agent playing.')
    args = parser.parse_args()
    agent = args.agent

    rand = False
    if agent == 'random':
        dir = './model/1'
        rand = True
    else:
        dir = {'1': './model/1', '2': './model/2', '3': './model/3'}[agent]

    sess = tf.Session()
    model = load(sess, dir, 'model.ckpt.meta')
    play_game(sess, model, rand)