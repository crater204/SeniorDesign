import datetime

import gym
import numpy as np
import pandas
import tensorflow as tf
import matplotlib.pylab as plt
import random
import math
import time

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.0001
GAMMA = 0.99
BATCH_SIZE = 50

hourList = []
occupantNumber1List = []
occupantNumber2List = []
indoorTempList = []
outdoorTempList = []
humidityList = []

class Model:
    def __init__(self, num_states, num_actions, batch_size):
        self._num_states = num_states
        self._num_actions = num_actions
        self._batch_size = batch_size
        # define the placeholders
        self._states = None
        self._actions = None
        # the output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None
        # now setup the model
        self._define_model()

    def _define_model(self):
        self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
        # create a couple of fully connected hidden layers
        fc1 = tf.layers.dense(self._states, 50, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 50, activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc2, self._num_actions)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()

    def predict_one(self, state, sess):
        reshapedState = state.reshape()
    #    print(type(state))
        return sess.run(self._logits, feed_dict={self._states: reshapedState})

    def predict_batch(self, states, sess):
 #       print("Batch")
        #print(states)
        newStates = []
        for s in states:
  #          print(s)
            reshaped = s.reshape()
         #   print(reshaped)
            newStates.append(reshaped[0])
        feed_dict={self._states: newStates}
  #      print(feed_dict)
        
        return sess.run(self._logits, feed_dict=feed_dict)

    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def var_init(self):
        return self._var_init


class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)


class GameRunner:
    nextIndex = 0

    def __init__(self, sess, model, env, memory, max_eps, min_eps,
                 decay):
        self._sess = sess
        self._env = env
        self._model = model
        self._memory = memory
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._decay = decay
        self._eps = self._max_eps
        self._steps = 0
        self._reward_store = []

    def run(self):
        state = self._env.reset()
        tot_reward = 0
        while True:
            action = self._choose_action(state)
            next_state, reward, done, info = self._env.step(action)

            self._memory.add_sample((state, action, reward, next_state))
            self._replay()

            # exponentially decay the eps value
            self._steps += 1
            self._eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) \
                                      * math.exp(-LAMBDA * self._steps)

            # move the agent to the next state and accumulate the reward
  #          state = next_state
            tot_reward += reward

            # if the game is done, break the loop
            if done:
                self._reward_store.append(tot_reward)
                break

        print("Step {}, Total reward: {}, Eps: {}".format(self._steps, tot_reward, self._eps))

    def _choose_action(self, state):
        if random.random() < self._eps:
            return random.randint(0, self._model.num_actions - 1)
        else:
            return np.argmax(self._model.predict_one(state, self._sess))

    def _replay(self):
        batch = self._memory.sample(self._model.batch_size)
        states = np.array([val[0] for val in batch])
        next_states = np.array([(np.zeros(self._model.num_states)
                                 if val[3] is None else val[3]) for val in batch])
        # predict Q(s,a) given the batch of states
   #     print(states)
        q_s_a = self._model.predict_batch(states, self._sess)
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self._model.predict_batch(next_states, self._sess)
        # setup training arrays
        x = np.zeros((len(batch), self._model.num_states))
        y = np.zeros((len(batch), self._model.num_actions))
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            # get the current q values for all actions in state
            current_q = q_s_a[i]
            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + GAMMA * np.amax(q_s_a_d[i])
    #        print(state)
    #        print(type(state))
            states = state.reshape()

            x[i] = states[0]
            y[i] = current_q
        self._model.train_batch(self._sess, x, y)

    @property
    def reward_store(self):
        return self._reward_store

class ThermalControlState:
    def __init__(self, i):
        self.index = i
#        if i == -1:
#            print(len(hourList))
#            self.index = random.randint(0, len(hourList) - 1)
        self.indoorTemp = indoorTempList[self.index]
        self.outdoorTemp = outdoorTempList[self.index]
        self.humidity = humidityList[self.index]
        self.time = hourList[self.index]
        self.occupied = max(occupantNumber1List[self.index], occupantNumber2List[self.index])
        self.score = 0

    def getStateArray(self):
        return [self.indoorTemp, self.outdoorTemp, self.humidity, self.time, self.occupied]

    def reshape(self):
        return [[self.indoorTemp, self.outdoorTemp, self.humidity, self.time, self.occupied]]

class ThermalControlEnv:

    def __init__(self):
        self.state = ThermalControlState(GameRunner.nextIndex)
        GameRunner.nextIndex += 1

    def reset(self):
        self.state = ThermalControlState(GameRunner.nextIndex)
        GameRunner.nextIndex += 1
        return self.state

    def step(self, action):
        reward = 0
        desiredAction = 0
        next_state = ThermalControlState(self.state.index + 1)
        if self.state.indoorTemp < next_state.indoorTemp:
            desiredAction = 1

        if action == desiredAction:
            reward = 1

        done = True
        info = {}
        return next_state, reward, done, info

if __name__ == "__main__":

    # Time, Occupant number, Occupied bool, Indoor Temp (degrees C)
   # data = pandas.read_excel("langevincodebook.xlsx", sheet_name="LANGEVIN_DATA", usecols=[0, 1, 3, 5], header=None)

    data = pandas.read_csv("office3_floor3_data.csv")
    print(data)
    print("Data Read Successful")
    matlabTimeList = data["Time"].to_list()
    print(matlabTimeList)

    for matlab_datenum in matlabTimeList:
        realtime = datetime.datetime.fromordinal(int(matlab_datenum)) + datetime.timedelta(
            days=matlab_datenum % 1) - datetime.timedelta(days=366)
        hourPart = realtime.hour
        minutePart = realtime.minute
        hour = hourPart + minutePart / 60
        hourList.append(hour)

    print("Time Converstion Succesful")
    occupantNumber1List = data["Occupancy 1"].to_list()
    occupantNumber2List = data["Occupancy 2"].to_list()
    indoorTempList = data["INDOOR Ambient Temp."].to_list()
    humidityList = data["INDOOR Relative Humidity"].to_list()
    outdoorTempList = data["OUTDOOR Ambient Temp."].to_list()

    print("Temp")

  #  EXCEL_DATA_ARRAY = [hourList, occupantNumberList, occupiedBoolList, indoorTempList]
    print("All Excel Successful")

    env_name = 'Thermal Control'
    env = ThermalControlEnv()

    num_states = 5 #temp, time, occupancy, outdoor temp, humidity
                   #or occupied, not occupied? so 2
    num_actions = 2

    model = Model(num_states, num_actions, BATCH_SIZE)
    mem = Memory(50000000)

    with tf.Session() as sess:
        sess.run(model.var_init)
        gr = GameRunner(sess, model, env, mem, MAX_EPSILON, MIN_EPSILON,
                        LAMBDA)
        # We take 1/4 of the year because this represents 1 season and hopefully
        # with similar weather (within a season) we will see more consistent data
        num_episodes = len(hourList) / 4
        cnt = 0
        while cnt < num_episodes:
            if cnt % 100 == 0:
                print('Episode {} of {}'.format(cnt+1, num_episodes))
            gr.run()
            cnt += 1

        # Break the reward store into a list of list that have 24 * 4 (1 day) elements each
        n = 24 * 4
        listOfLists = [gr.reward_store[i * n:(i + 1) * n] for i in range((len(gr.reward_store) + n - 1) // n)]
        rewardSums = []
        index = 0
        for innerList in listOfLists:
            rewardSums.append(0)
            for reward in innerList:
                rewardSums[index] += reward
            index += 1
        # remove the last element because the last list in listOfLists won't have 100 elements
        rewardSums = rewardSums[:-1]

        plt.plot(rewardSums, "bo", markersize=3)
        plt.xlabel("Days")
        plt.ylabel("Correct Predictions (96 attempted each day)")
        plt.title("Reinforcement learning over 3 Months")
        plt.show()
        plt.close("all")
