import pandas as pd
import numpy as np
import sys
import random


STATE_SIZE = 20
BATCH_SIZE = 100
UPDATE_TARGET_INTERVAL = 100
GAMMA = 0.79
LEARNING_RATE = 0.001
TESTING_LEARNING_RATE = 0.001

# Read in dataset
data = pd.read_csv('EURUSD-2005-2020.csv')



# Retrieve states
states = []
upper_index = 19
for day in range(upper_index, len(data)):
    states.append(np.array(data.loc[day-upper_index:day, 'Adj_Close']))
states = np.array(states)

# Calculate the split indices
two_thirds = int(len(states) * 2/3)

# Split the array
training_set = states[:two_thirds]
testing_set = states[two_thirds:]

# Create Dense Layer (AND TEST!!!)
class Dense():
    def __init__(self, input_neurons, output_neurons, name):
        self.name = name
        self.weights = np.random.normal(size=(input_neurons, output_neurons), loc=0, scale=np.sqrt(2/input_neurons))
        self.biases = np.zeros(shape=(output_neurons))
    def forward(self, input):
            self.input = input
            return np.add(np.dot(input, self.weights), self.biases)
    def backward(self, de_dy, learning_rate): 
            de_db = de_dy
            de_dw = np.dot(de_dy.T, np.mean(self.input, axis=0)).T
            de_dx = np.dot(de_dy, self.weights.T)
            self.weights = np.subtract(self.weights, np.multiply(learning_rate, de_dw))
            self.biases = np.subtract(self.biases, np.multiply(learning_rate, de_db))
            return de_dx
    


# Create ReLU layer (AND TEST)
class ReLU():
    def __init__(self, name):
        self.name = name
    def forward(self, input):
        self.input = input
        return np.maximum(0, self.input)
    def backward(self, de_dy, learning_rate):
        de_dx = np.where(self.input >= 0, np.repeat(np.expand_dims(de_dy, axis=0), self.input.shape[0], axis=0), 0)
        return np.mean(de_dx, axis=0)

network = [
    Dense(20,10,"D1"),
    ReLU("R1"),
    Dense(10,10,"D2"),
    ReLU("R2"),
    Dense(10,3,"D3")
]


# Create Loss Class
class MSE:
    def __init__(self, name):
        self.name = name
    def forward(self, prediction, target):
        loss = np.mean(np.power(prediction - target), 2)
        return loss
    def backward(self, prediction, target): # May need to revisit
        return 2 * (prediction - target) / np.size(prediction)
mean_squared_error = MSE("MSE1")





# Create DQN class (AND TEST)
class DQN():
    def __init__(self, network, loss):
        self.network = network
        self.loss = loss
    def forward(self, x):
        for layer in self.network:
            x = layer.forward(x)
        return self.loss.forward(x)
    def backward(self):
        x = self.loss.backward()
        for layer in self.network[::-1]:
            x = layer.backward(x)
    def query(self, x):
        for layer in self.network:
           x = layer.forward(x)
        return x
    def backprop(self, prediction, target, learning_rate):
        l = self.loss.forward(prediction, target)
        x = self.loss.backward(prediction, target)
        for layer in self.network[::-1]:
            x = layer.backward(x, learning_rate)
        return l
    def retrieve_network(self):
        return self.network
    def set_network(self, network):
        self.network = network
trading_dqn = DQN(network=network, loss=mean_squared_error)


# Create Replay Buffer (AND TEST)
class Replay_Buffer():
    def __init__(self, size):
        self.next_index = 0
        self.size = size
        self.states = np.zeros(shape=(size, STATE_SIZE))
        self.actions = np.zeros(shape=(size, 1))
        self.rewards = np.zeros(shape=(size, 1))
        self.states_primes = np.zeros(shape=(size, STATE_SIZE))   
    def sample(self, batch_size):
        sample_indices = np.random.randint(0, self.size, size=batch_size)
        selected_states = self.states[sample_indices]
        selected_actions = self.actions[sample_indices]
        selected_rewards = self.rewards[sample_indices]
        selected_state_primes = self.states_primes[sample_indices]
        return selected_states, selected_actions, selected_rewards, selected_state_primes
    def add(self, state, action, reward, state_prime):
        self.states[self.next_index] = state
        self.actions[self.next_index] = action
        self.rewards[self.next_index] = reward
        self.states_primes[self.next_index] = state_prime
        self.next_index+=1
        if self.next_index == self.size:
            self.next_index = 0
trades_replay_buffer = Replay_Buffer(size=3000)



# Create Environement
class Environment():
    def __init__(self, environment_states):
        self.environment_states = environment_states
        self.current_state_index = 0
        self.final_state_index = len(environment_states)-2
        self.terminal = False
    def step(self, actions):
        num_actions = len(actions)
        current_state = self.environment_states[self.current_state_index]
        next_state = self.environment_states[self.current_state_index+1]
        states = np.zeros(shape=(num_actions, STATE_SIZE))
        state_primes = np.zeros(shape=(num_actions, STATE_SIZE))
        rewards = np.zeros(shape=(num_actions,1))
        for index, action in enumerate(actions):
            states[index] = current_state
            rewards[index] = (next_state[STATE_SIZE-1]-current_state[STATE_SIZE-1]) * action
            state_primes[index] = next_state
        if self.current_state_index == self.final_state_index:
            self.terminal = True
        else:
            self.current_state_index +=1
        return (states, actions, rewards, state_primes)
    def reset(self):
        self.terminal = False
        self.current_state = 0
training_environment = Environment(environment_states=training_set)
testing_environment = Environment(environment_states=testing_set)

    


# Create Agent
class Agent(object):
    def __init__(self, dqn_architecture, replay_buffer, training_batch_size, update_target_interval, training_learning_rate, testing_learning_rate):
        self.dqn = dqn_architecture
        self.target_dqn = dqn_architecture
        self.replay_buffer = replay_buffer
        self.training_batch_size = training_batch_size
        self.update_target_interval = update_target_interval
        self.training_learning_rate = training_learning_rate
        self.testing_learning_rate = testing_learning_rate
        self.epsilon = 1.0
    def train(self, environment, num_iterations):
        iteration_num = 0
        # Epsilon
        epsilon_decrement_per_step = 1/((len(environment.environment_states)-1)*num_iterations)
        # fill replay_buffer
        legal_actions = [-1,0,1]
        for i in range(int(trades_replay_buffer.size/3)+1):
            states, actions, rewards, state_primes = environment.step(legal_actions)
            for i in range(len(states)):
                trades_replay_buffer.add(state=states[i], action=actions[i], reward=rewards[i], state_prime=state_primes[i])
        for i in range(num_iterations):
            environment.reset()
            while True:
                if environment.terminal == True:
                    break
                q_values = self.dqn.query(environment.environment_states[environment.current_state_index])
                random_value = random.random()
                action = -2
                if random_value<self.epsilon:
                    action = [np.random.choice(legal_actions)]
                else:
                    action = [legal_actions[np.argmax(q_values)]]
                self.epsilon -= epsilon_decrement_per_step
                states, actions, rewards, state_primes = environment.step(action)
                for i in range(len(states)):
                    trades_replay_buffer.add(state=states[i], action=actions[i], reward=rewards[i], state_prime=state_primes[i])
                selected_states, selected_actions, selected_rewards, selected_state_primes = trades_replay_buffer.sample(BATCH_SIZE)
                predictions = np.zeros(shape=(BATCH_SIZE,))
                targets = np.zeros(shape=(BATCH_SIZE,))
                for i in range(BATCH_SIZE): # use the online model instead of the target model when selectin the best actions in the next states
                    target_q_value_index = np.argmax(self.dqn.query(selected_state_primes[i]))
                    target_q_value = self.target_dqn.query(selected_state_primes[i])[target_q_value_index]
                    target = selected_rewards[i] + GAMMA * target_q_value
                    np.append(targets, target)
                    returned_q_values = self.dqn.query(selected_states[i])
                    chosen_q_value = returned_q_values[np.where(legal_actions == selected_actions[i])[0]]
                    np.append(predictions, chosen_q_value)
                self.dqn.backprop(predictions, targets, self.training_learning_rate)
                self.target_dqn.set_network(self.dqn.retrieve_network())
                self.training_learning_rate = LEARNING_RATE * (0.1 ** (iteration_num/self.update_target_interval))
                iteration_num += 100

    def evaluate(self, environment):
        pass
    def save(self):
        pass
    def load(self):
        pass
trading_agent = Agent(dqn_architecture=trading_dqn, replay_buffer=trades_replay_buffer, training_batch_size=BATCH_SIZE, update_target_interval=UPDATE_TARGET_INTERVAL, training_learning_rate=LEARNING_RATE, testing_learning_rate=TESTING_LEARNING_RATE)
trading_agent.train(environment=training_environment, num_iterations=5)