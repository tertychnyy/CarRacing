import random

import gym
import numpy as np
import tensorflow as tf
from scipy.misc import imsave


# These are from the TensorFlow tutorial to initalize weights and variable with a slight
# positive bias to prevent dead neurons
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Convolutional and Pooling functions also from TensorFlow Tutorials
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# HYPERPARMETERS
H = 100
H2 = 100
batch_number = 50
batch_size = 60
gamma = 0.99
explore = 1
num_of_episodes_between_q_copies = 50
learning_rate = 1e-3


if __name__ == '__main__':

    env = gym.make('CarRacing-v0')
    print "Gym input is ", env.action_space
    print "Gym observation is ", env.observation_space

    # Setup tensorflow
    shape1 = env.observation_space.shape[0]
    shape2 = env.observation_space.shape[1]
    shape3 = env.observation_space.shape[2]

    output_shape = env.action_space.shape[0]
    print output_shape

    tf.reset_default_graph()

    # First Q Network
    images = tf.placeholder(tf.float32, [None, shape1, shape2, shape3])

    W_conv1 = weight_variable([5, 5, 3, 25])
    b_conv1 = bias_variable([25])

    h_conv1 = tf.nn.relu(conv2d(images, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    #W_conv2 = weight_variable([5, 5, 25, 50])
    #b_conv2 = bias_variable([50])

    #h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    #W_fc1 = weight_variable([28800, 1024])
    W_fc1 = weight_variable([57600, 1024])
    b_fc1 = bias_variable([1024])

    #h_pool2_flat = tf.reshape(h_pool2, [-1, 28800])
    h_pool2_flat = tf.reshape(h_pool1, [-1, 57600])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, output_shape])
    b_fc2 = bias_variable([output_shape])

    Q = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


    # Second Q Network
    images_ = tf.placeholder(tf.float32, [None, shape1, shape2, shape3])

    W_conv1_ = weight_variable([5, 5, 3, 25])
    b_conv1_ = bias_variable([25])

    h_conv1_ = tf.nn.relu(conv2d(images_, W_conv1_) + b_conv1_)
    h_pool1_ = max_pool_2x2(h_conv1_)

    #W_conv2_ = weight_variable([5, 5, 25, 50])
    #b_conv2_ = bias_variable([50])

    #h_conv2_ = tf.nn.relu(conv2d(h_pool1_, W_conv2_) + b_conv2_)
    #h_pool2_ = max_pool_2x2(h_conv2_)

    #W_fc1_ = weight_variable([28800, 1024])
    W_fc1_ = weight_variable([57600, 1024])
    b_fc1_ = bias_variable([1024])

    #h_pool2_flat_ = tf.reshape(h_pool2_, [-1, 28800])
    h_pool2_flat_ = tf.reshape(h_pool1_, [-1, 57600])
    h_fc1_ = tf.nn.relu(tf.matmul(h_pool2_flat_, W_fc1_) + b_fc1_)

    keep_prob_ = tf.placeholder(tf.float32)
    h_fc1_drop_ = tf.nn.dropout(h_fc1_, keep_prob_)

    W_fc2_ = weight_variable([1024, output_shape])
    b_fc2_ = bias_variable([output_shape])

    Q_ = tf.matmul(h_fc1_drop_, W_fc2_) + b_fc2_


    # need to run these to assign weights from Q to Q_prime

    W_conv1_update = W_conv1_.assign(W_conv1)
    b_conv1_update = b_conv1_.assign(b_conv1)

    #W_conv2_update = W_conv2_.assign(W_conv2)
    #b_conv2_update = b_conv2_.assign(b_conv2)

    W_fc1_update = W_fc1_.assign(W_fc1)
    b_fc1_update = b_fc1_.assign(b_fc1)

    W_fc2_update = W_fc2_.assign(W_fc2)
    b_fc2_update = b_fc2_.assign(b_fc2)

    assign_all = [
        W_conv1_update,
        b_conv1_update,
        #W_conv2_update,
        #b_conv2_update,
        W_fc1_update,
        b_fc1_update,
        W_fc2_update,
        b_fc2_update]

    # we need to train Q

    rewards = tf.placeholder(tf.float32, [None, ],
                             name="rewards")  # This holds all the rewards that are real/enhanced with Qprime
    actions = tf.placeholder(tf.int32, [None], name="training_mask")
    one_hot_actions = tf.one_hot(actions, 3)
    Q_filtered = tf.reduce_sum(tf.multiply(Q, one_hot_actions), reduction_indices=1)
    loss = tf.reduce_sum(tf.square(rewards - Q_filtered))  # * one_hot
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Setting up the enviroment

    max_episodes = 20000
    max_steps = 2000

    D = []
    explore = 1.0

    rewardList = []
    past_actions = []

    episode_number = 0
    episode_reward = 0
    reward_sum = 0

    init = tf.initialize_all_variables()


    with tf.Session() as sess:
        sess.run(init)
        # Copy Q over to Q_prime
        sess.run(assign_all)

        for episode in xrange(max_episodes):
            print 'Reward for episode %f is %f. Explore is %f' % (episode, reward_sum, explore)

            reward_sum = 0
            new_state = env.reset()

            for step in xrange(max_steps):
                #if step % batch_number == 0:
                #    env.render()
                #    print step

                env.render()

                state = list(new_state)

                if explore > random.random():
                    action_sample = env.action_space.sample()
                    action = np.argmax(action_sample)
                else:
                    # get action from policy
                    results = sess.run(action_values, feed_dict={states: np.array([new_state]), keep_prob_: 1})
                    # print results
                    action = (np.argmax(results[0]))

                curr_action = action

                action_temp = [0.0, 1.0, 0.0]
                # action_temp[action] = 1.0
                new_state, reward, done, _ = env.step(action_temp)
                reward_sum += reward

                D.append([state, curr_action, reward, new_state, done])

                if len(D) > 5000:
                    D.pop(0)

                # Training a Batch
                # samples = D.sample(50)
                sample_size = len(D)
                if sample_size > 500:
                    sample_size = 500
                else:
                    sample_size = sample_size

                if True:
                    samples = [D[i] for i in random.sample(xrange(len(D)), sample_size)]
                    # print samples
                    new_states_for_q = [x[3] for x in samples]
                    # print new_states_for_q
                    all_q_prime = sess.run(Q_, feed_dict={images_: new_states_for_q, keep_prob_: 1})
                    # print "All The Q Primes:", all_q_prime
                    y_ = []
                    states_samples = []
                    next_states_samples = []
                    actions_samples = []
                    for ind, i_sample in enumerate(samples):
                        # print i_sample
                        if i_sample[4] == True:
                            # print i_sample[2]
                            y_.append(reward)
                            # print y_
                        else:
                            this_q_prime = all_q_prime[ind]
                            maxq = max(this_q_prime)
                            y_.append(reward + (gamma * maxq))
                            # print y_
                        # y_.append(i_sample[2])
                        states_samples.append(i_sample[0])
                        next_states_samples.append(i_sample[3])
                        actions_samples.append(i_sample[1])

                    sess.run(train,
                             feed_dict={images: states_samples, rewards: y_, keep_prob: .7, actions: actions_samples})

                    if done:
                        break

            if episode % num_of_episodes_between_q_copies == 0:
                sess.run(assign_all)

            explore = explore * .9997
