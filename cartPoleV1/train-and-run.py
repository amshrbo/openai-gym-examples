import tensorflow as tf
import numpy as np
import gym
import gym.spaces

#Creating our network variables
num_inputs = 4 # our observations
num_hidden = 4
num_outputs = 1 # probability to go left --> 1-left = right
learning_rate = 0.01
initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, num_inputs])

hid_lyr = tf.layers.dense(X, num_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
# get the outputs doing it in two seprate steps because we are gonna use the two vars
logits = tf.layers.dense(hid_lyr, num_outputs)
outputs = tf.nn.sigmoid(logits)

# after we applaying the sigmoid
probabilities = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(probabilities, num_samples=1)

# get some sort of output y to train the network on
y = 1.0 - tf.to_float(action)

#loss function and optimization 
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
# instead of minimize our loss function we are going to calculate the gradients
# later on we are gonna to multiply these by discount rate
gradients_and_vars = optimizer.compute_gradients(cross_entropy) #returns to variable the var and its gradient

gradients = []
gradient_placeholders = []
grads_and_vars_feed = []
# tuple unpacking
for gradient, variable in gradients_and_vars:
    gradients.append(gradient) # fill out the gradients list
    # create a placeholder for every gradient
    gradient_placeholder = tf.placeholder(tf.float32, shape=gradient.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))

training_op = optimizer.apply_gradients(grads_and_vars_feed)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

def helper_discount_reward(rewards, discount_rate):
    '''
    takes in some rewards and applies discount rate on them
    '''
    discounted_rewards = np.zeros(len(rewards)) # initiating the list
    cumulative_rewards = 0
    # reversed iterate over the sequence reversed e.g form 10 -> 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    '''
    takes in all the rewards applaying helper_discount functionand then normalize the values
    using mean and std. the time it takes to achieve this function increases when we playing
    more games because of the number of all_rewards increased
    '''
    all_discounted_rewards = []
    for rewards in all_rewards:
        all_discounted_rewards.append(helper_discount_reward(rewards, discount_rate))
    # applaying normalization
    flatten_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flatten_rewards.mean()
    reward_std = flatten_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]

# Creating our environment
env = gym.make('CartPole-v0')
num_game_rounds = 30
max_game_steps = 1000
num_iterations = 400
discount_rate = 0.9

with tf.Session() as sess:
    sess.run(init)

    for iteration in range(num_iterations):
        print("On Iteration: {}".format(iteration))
        all_rewards = []
        all_gradients = []

        for game in range(num_game_rounds):
            current_rewards = []
            current_gradients = []
            observations = env.reset()

            for step in range(max_game_steps):
                action_val, gradients_val = sess.run([action, gradients], feed_dict={X: observations.reshape(1, num_inputs)})
                
                observations, reward, done, info = env.step(action_val[0][0])
                current_rewards.append(reward)
                current_gradients.append(gradients_val)
                # chk if the pole gets down or something like that
                if done:
                    break
            
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)
        # applaying discount rate and normalization 
        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
        feed_dict = {}

        #Applaying the scores to the gradients
        # enumerate just gave back index location with every val
        for var_index, gradient_placeholder in enumerate(gradient_placeholders):
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                      for game_index, rewards in enumerate(all_rewards)
                                        for step, reward in enumerate(rewards)], axis=0)
            feed_dict[gradient_placeholder] = mean_gradients
        sess.run(training_op, feed_dict=feed_dict)

        print("Saving graph and session")
        meta_graph = tf.train.export_meta_graph(filename='./savedModel/myPolicyModel2.meta')
        saver.save(sess, './savedModel/myPolicyModel2')


# Running our trained model
env = gym.make('CartPole-v0')
obs = env.reset()
with tf.Session() as ses:
    saver = tf.train.import_meta_graph('./savedModel/myPolicyModel2.meta')
    saver.restore(ses, './savedModel/myPolicyModel2')

    for x in range(500):
        env.render()
        action_val, gradients_val = ses.run([action, gradients], feed_dict={X: obs.reshape(1, num_inputs)})
        obs, reward, done, info = env.step(action_val[0][0])