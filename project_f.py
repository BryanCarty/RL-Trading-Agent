import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import numpy as np
import sys
import random
import tensorboard as tb


# Constants
STATE_SIZE = 20 #  20 states used in prediction
BATCH_SIZE = 100 # 100 samples trained at a time
GAMMA = 0.95 # Care more about immediate rewards
#LEARNING_RATE = 0.00001
#LEARNING_RATE = 0.00001 # too low - not switching at all


LEARNING_RATE = 0.00001
#LEARNING_RATE = 0.001 # too high - switching wholely
NUM_ACTIONS = 3


tensorboard_writer = tf.summary.create_file_writer('tensorboard/')


# Obtain States
data = pd.read_csv('EURUSD-2005-2020.csv')
states = []
upper_index = 19
for day in range(upper_index, len(data)):
    states.append(np.array((data.loc[day-upper_index:day, 'Adj_Close'])*100))
states = np.array(states)
three_fourths = int(len(states) * 3/4)
training_set = states[:three_fourths]
testing_set = states[three_fourths:]


# Validate data
print("Training Set Length: " + str(len(training_set)))
print("Testing Set Length: " + str(len(testing_set)))
for i in training_set:
    if len(i) != 20:
        sys.exit('Training Data: Invalid Length Encountered!')
for i in testing_set:
    if len(i) != 20:
        sys.exit('Testing Data: Invalid Length Encountered!')

# Create Replay Buffer
class Replay_Buffer():
    def __init__(self, size):
        self.next_index = 0
        self.size = size
        self.states = np.zeros(shape=(size, STATE_SIZE))
        self.actions = np.zeros(shape=(size,))
        self.rewards = np.zeros(shape=(size,))
        self.states_primes = np.zeros(shape=(size, STATE_SIZE))   
        self.terminal_states = np.zeros(shape=(size,))
    def sample(self, batch_size):
        sample_indices = np.random.randint(0, self.size, size=batch_size) # assumes buffer has been filled first -> At some point may need to ensure they're not repeating
        selected_states = self.states[sample_indices]
        selected_actions = self.actions[sample_indices]
        selected_rewards = self.rewards[sample_indices]
        selected_state_primes = self.states_primes[sample_indices]
        selected_terminals = self.terminal_states[sample_indices]
        return selected_states, selected_actions, selected_rewards, selected_state_primes, selected_terminals
    def add(self, state, action, reward, state_prime, terminal):
        self.states[self.next_index] = state
        self.actions[self.next_index] = action
        self.rewards[self.next_index] = reward
        self.states_primes[self.next_index] = state_prime
        self.terminal_states[self.next_index] = terminal
        self.next_index+=1
        if self.next_index == self.size:
            self.next_index = 0
trades_replay_buffer = Replay_Buffer(size=10000) # Initially there's 3115 * 3 = 9345 unique samples




# Create Environement
class Environment(): # the agent will receive the terminal state when the running capital goes below 30% of initial capital or if the environment reaches the end of time series training data.
    def __init__(self, environment_states):
        self.environment_states = environment_states
        self.current_state_index = 0
        self.final_state_index = len(environment_states)-1
        self.terminal = False
        self.acccumulated_reward = None
    def step(self, actions):
        num_actions = len(actions)
        if self.acccumulated_reward is None:
            self.acccumulated_reward = np.zeros(shape=(num_actions,))
        current_state = self.environment_states[self.current_state_index]
        states = np.zeros(shape=(num_actions, STATE_SIZE))
        if self.current_state_index == self.final_state_index:
            self.terminal = True
            for index, val in enumerate(self.acccumulated_reward):
                if val < 0:
                    self.acccumulated_reward[index] = val
                else:
                    self.acccumulated_reward[index] = val
                states[index] = current_state                
            return (states, actions, self.acccumulated_reward, np.zeros(shape=(states.shape)), np.ones(shape=(len(self.acccumulated_reward))))
        rewards = np.zeros(shape=(num_actions,))
        next_state = self.environment_states[self.current_state_index+1]
        state_primes = np.zeros(shape=(num_actions, STATE_SIZE))
        action_values = [-1,0,1]
        for index, action in enumerate(actions):
            states[index] = current_state
            rewards[index] = ((next_state[STATE_SIZE-1]-current_state[STATE_SIZE-1]) * action_values[action])
            self.acccumulated_reward[index] += rewards[index]
            state_primes[index] = next_state
        self.current_state_index +=1
        return (states, actions, rewards, state_primes, np.zeros(shape=(num_actions,)))
    def reset(self):
        self.terminal = False
        self.current_state_index = 0
        self.acccumulated_reward = None
training_environment = Environment(environment_states=training_set)
testing_environment = Environment(environment_states=testing_set)

# DQN
dqn = tf.keras.Sequential()
dqn.add(tf.keras.layers.Dense(units=32, activation="relu", input_shape=(1,STATE_SIZE)))
dqn.add(tf.keras.layers.Dense(units=64, activation="relu"))
dqn.add(tf.keras.layers.Dense(units=128, activation="relu"))
dqn.add(tf.keras.layers.Dense(units=NUM_ACTIONS, activation="linear"))
dqn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),loss=tf.keras.losses.Huber())

# Target DQN
tdqn = tf.keras.Sequential()
tdqn.add(tf.keras.layers.Dense(units=32, activation="relu", input_shape=(1,STATE_SIZE)))
tdqn.add(tf.keras.layers.Dense(units=64, activation="relu"))
tdqn.add(tf.keras.layers.Dense(units=128, activation="relu"))
tdqn.add(tf.keras.layers.Dense(units=NUM_ACTIONS))
tdqn.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))



# Create Agent
class Agent(object):
    def __init__(self, dqn_architecture, target_dqn_architecture, replay_buffer):
        self.dqn = dqn_architecture
        self.target_dqn = target_dqn_architecture
        self.replay_buffer = replay_buffer
        self.epsilon = 1.0
    def train(self, environment, num_iterations, testing_environment):
        self.iter_num = 0
        # Epsilon - Do we need epsilon if the state space is small?
        self.total_iterations = len(environment.environment_states)*num_iterations
        epsilon_decrement_per_step = 1/self.total_iterations
        # fill replay_buffer
        legal_actions = [0,1,2]
        for i in range(int(trades_replay_buffer.size/3)+1):
            if environment.terminal == True:
                environment.reset()
            states, actions, rewards, state_primes, terminals = environment.step(legal_actions) # fill up buffer with epsilon = 1.0 initially
            for i in range(len(states)):
                trades_replay_buffer.add(state=states[i], action=actions[i], reward=rewards[i], state_prime=state_primes[i], terminal=terminals[i])
        for i in range(num_iterations):
            environment.reset()
            while True:
                if environment.terminal == True:
                    break
                input_state = tf.convert_to_tensor(np.reshape(environment.environment_states[environment.current_state_index], newshape=(1, 1, STATE_SIZE)), dtype=tf.float32)
                q_values = self.dqn(input_state, training=False).numpy()
                random_value = random.random()
                action = -1
                if random_value<self.epsilon:
                    action = [random.randint(0, 2)]
                else:
                    action = [np.argmax(q_values)]
                self.epsilon -= epsilon_decrement_per_step
                states, actions, rewards, state_primes, terminals = environment.step(action)
                trades_replay_buffer.add(state=states[0], action=actions[0], reward=rewards[0], state_prime=state_primes[0], terminal=terminals[0])
                
                # Sample from replay buffer
                selected_states, selected_actions, selected_rewards, selected_state_primes, selected_terminals = trades_replay_buffer.sample(BATCH_SIZE) 
                next_states_tensor = tf.convert_to_tensor(np.reshape(selected_state_primes, newshape=(BATCH_SIZE, 1, STATE_SIZE)), dtype=tf.float32) #convert next states to tf.tensor of correct shape
                query_results = self.dqn(next_states_tensor, training=False).numpy() 
                best_action_in_next_state_using_main_dqn = query_results.argmax(axis=2) # shape = (100,1)
                if self.iter_num % 5000 == 0:
                    print(best_action_in_next_state_using_main_dqn)
                target_q_network_q_values = np.reshape(self.target_dqn(next_states_tensor, training=False).numpy(), newshape=(BATCH_SIZE, NUM_ACTIONS)) # shape = (100,3)
                optimal_q_value_in_next_state_target_dqn = np.zeros(shape=(100,)) # Why are the optimal Q-Values the same across the batch
                for i, v in enumerate(target_q_network_q_values):
                    optimal_q_value_in_next_state_target_dqn[i] = v[best_action_in_next_state_using_main_dqn[i][0]]
                target_q_values = tf.convert_to_tensor(np.add(selected_rewards, np.multiply(GAMMA, optimal_q_value_in_next_state_target_dqn) * (1 - selected_terminals)), dtype=tf.float32)
                with tf.GradientTape() as tape:
                    q_values_current_state_dqn = tf.reshape(self.dqn(tf.convert_to_tensor(np.reshape(selected_states, newshape=(BATCH_SIZE, 1, STATE_SIZE)), dtype=tf.float32), training=False), shape=(BATCH_SIZE, NUM_ACTIONS))
                    one_hot_actions = tf.keras.utils.to_categorical(selected_actions, NUM_ACTIONS, dtype=np.float32) 
                    Q = tf.reduce_sum(tf.multiply(q_values_current_state_dqn, one_hot_actions), axis=1)
                    loss = tf.keras.losses.Huber()(target_q_values, Q)
                dqn_architecture_gradients = tape.gradient(loss, self.dqn.trainable_variables) 
                self.dqn.optimizer.apply_gradients(grads_and_vars=zip(dqn_architecture_gradients, self.dqn.trainable_variables))  
                if self.iter_num % 1500 == 0:
                    self.target_dqn.set_weights(self.dqn.get_weights())
                if self.iter_num % 100 == 0:
                    with tensorboard_writer.as_default():
                        tf.summary.scalar('REWARD', np.mean(selected_rewards), self.iter_num)
                        tf.summary.scalar('LOSS', loss, self.iter_num)
                        #self.evaluate(testing_environment)
                if self.iter_num % 2500 == 0:
                    print('Progress: '+str((self.iter_num/self.total_iterations)*100)+'%')
                self.iter_num += 1
    def evaluate(self, environment):
        with tensorboard_writer.as_default():
            reward_record = []
            percentage_gain_loss_per_day = []
            environment.reset()
            action_values = [-1,0,1]
            while True:
                if environment.terminal == True:
                    break
                input_state = tf.convert_to_tensor(np.reshape(environment.environment_states[environment.current_state_index], newshape=(1, 1, STATE_SIZE)), dtype=tf.float32)
                q_values = self.dqn(input_state, training=False).numpy()
                action = [np.argmax(q_values)]
                states, actions, rewards, state_primes = environment.step(action)
                reward_record.append(rewards[0])
                entry_price = states[0][len(states[0])-1]
                exit_price = state_primes[0][len(state_primes[0])-1]
                percentage_gain_loss = (((exit_price-entry_price)/entry_price)*100)*action_values[action[0]]
                percentage_gain_loss_per_day.append(percentage_gain_loss)
            avg_reward = np.mean(np.array(reward_record))
            avg_percentage_gain_loss = np.mean(np.array(percentage_gain_loss_per_day))
            tf.summary.scalar('AVERAGE REWARD', avg_reward, self.iter_num)
            tf.summary.scalar('AVERAGE PERCENTAGE GAIN/LOSS PER DAY', avg_percentage_gain_loss, self.iter_num)
    def save(self):
        pass
    def load(self):
        pass


trading_agent = Agent(dqn_architecture=dqn, target_dqn_architecture=tdqn, replay_buffer=trades_replay_buffer)
trading_agent.train(training_environment, 250, testing_environment)