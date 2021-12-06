from memory_remote import ReplayBuffer_remote
from dqn_model import DQNModel
import torch
import ray
import os
from random import uniform, randint
from custom_cartpole import CartPoleEnv
from copy import deepcopy
import time
import matplotlib.pyplot as plt
%matplotlib inline


@ray.remote
class DQNAgent_server(object):
    def __init__(self, env, hyper_params, action_space=2):

        self.collector_done = False
        self.evaluator_done = False

        self.env = env
        self.max_episode_steps = env._max_episode_steps

        self.beta = hyper_params['beta']
        self.initial_epsilon = 1
        self.final_epsilon = hyper_params['final_epsilon']
        self.epsilon_decay_steps = hyper_params['epsilon_decay_steps']

        self.episode = 0
        self.steps = 0
        self.best_reward = 0
        self.learning = True
        self.action_space = action_space

        state = env.reset()
        input_len = len(state)
        output_len = action_space
        self.eval_model = DQNModel(input_len, output_len, learning_rate=hyper_params['learning_rate'])
        self.use_target_model = hyper_params['use_target_model']
        if self.use_target_model:
            self.target_model = DQNModel(input_len, output_len)
        self.memory = ReplayBuffer_remote(hyper_params['memory_size'])

        self.batch_size = hyper_params['batch_size']
        self.update_steps = hyper_params['update_steps']
        self.model_replace_freq = hyper_params['model_replace_freq']

    # evalutor
    def add_result(self, result, num):
        self.results[num] = result

    def get_results(self):
        return self.results

    def ask_evaluation(self):
        if len(self.results) > self.result_count:
            num = self.result_count
            self.result_count += 1
            return self.eval_model, False, num
        else:
            if self.episode >= self.learning_episodes:
                self.evaluator_done = True
            return None, self.evaluator_done, None

    def greedy_policy(self, state):
        return self.eval_model.predict(state)

    def update(self):
        if len(self.memory) < self.batch_size or self.steps % self.update_steps != 0:
            return

        batch = ray.get(self.memory.sample.remote(self.batch_size))

        (states, actions, reward, next_states, is_terminal) = batch

        states = states
        next_states = next_states
        terminal = torch.FloatTensor([1 if t else 0 for t in is_terminal])
        reward = torch.FloatTensor(reward)
        batch_index = torch.arange(self.batch_size, dtype=torch.long)

        # Current Q Values
        _, q_values = self.eval_model.predict_batch(states)
        q_values = q_values[batch_index, actions]

        # Calculate target
        if self.use_target_model:
            actions, q_next = self.target_model.predict_batch(next_states)
        else:
            actions, q_next = self.eval_model.predict_batch(next_states)

        with torch.no_grad():
            q_target = reward + self.beta * q_next[batch_index, actions] * (1 - terminal)

        # update model
        self.eval_model.fit(q_values, q_target)

        if self.episode >= self.learning_episodes:
            self.collector_done = True

        return

    def episode_complete(self):
        self.episode += 1
        if self.episode >= self.learning_episodes:
            self.collector_done = True
        return self.collector_done


@ray.remote
def collecting_worker(model_server, memory_server, simulator, update_steps=10, max_episode_steps=200, action_space=2,
                      initial_epsilon=1, final_epsilon=0.1, epsilon_decay_steps=100000):

    def linear_decrease(initial_value, final_value, curr_steps, final_decay_steps):
        decay_rate = curr_steps / final_decay_steps
        if decay_rate > 1:
            decay_rate = 1
        return initial_value - (initial_value - final_value) * decay_rate

    def explore_or_exploit_policy(state):
        p = uniform(0, 1)
        epsilon = linear_decrease(initial_epsilon, final_epsilon, steps, epsilon_decay_steps)

        if p < epsilon:
            return randint(0, action_space - 1)
        else:
            return ray.get(model_server.greedy_policy(state))

    global_steps = 0
    learn_done = False

    while True:
        if learn_done:
            break

        state = simulator.reset()
        done = False
        steps = 0

        while steps < max_episode_steps and not done and not learn_done:
            action = explore_or_exploit_policy(state)
            next_state, reward, done, _ = simulator.step(action)
            memory_server.add.remote(state, action, reward, next_state, done)
            state = next_state
            steps += 1
            global_steps += 1

            if global_steps % update_steps == 0:
                model_server.update.remote()

        learn_done = ray.get(model_server.episode_complete.remote())


@ray.remote
def evaluation_worker(model_server, simulator, trials=100, action_space=2, max_episode_steps=200,
                      input_len=100, output_len=100):

    def greedy_policy(eval_model, state):
        return eval_model.predict(state)

    eval_model = DQNModel(input_len, output_len)

    while True:
        ref_model, done, num = ray.get(model_server.ask_evaluation.remote())

        if done:
            break
        if ref_model is None:
            continue

        eval_model = eval_model.replace(ref_model)

        total_reward = 0
        for _ in range(trials):
            state = simulator.reset()
            done = False
            steps = 0

            while steps < max_episode_steps and not done:
                state, reward, done, _ = simulator.step(greedy_policy(eval_model, state))
                total_reward += reward
                steps += 1

        avg_reward = total_reward / trials
        model_server.add_results.remote(avg_reward, num)


class distributed_DQN_agent():
    def __init__(self, env, hyper_params, cw_num=4, ew_num=4, test_interval=100,
                 action_space=2, do_test=True):

        self.env = env
        self.model_server = DQNAgent_server.remote(env, hyper_params, action_space)
        self.workers_id = []
        self.cw_num = cw_num
        self.ew_num = ew_num
        self.agent_name = "Distributed DQN training"
        self.do_test = do_test

    def learn_and_evaluate(self):
        workers_id = []

        for i in range(self.cw_num):
            w_id = collecting_worker.remote(self.server, deepcopy(self.env), self.epsilon)
            workers_id.append(w_id)

        if self.do_test:
            for i in range(self.ew_num):
                w_id = evaluation_worker.remote(self.server, deepcopy(self.env))
                workers_id.append(w_id)

        ray.wait(workers_id, num_returns=len(workers_id), timeout=None)
        return ray.get(self.server.get_results.remote())


def plot_result(total_rewards, learning_num, legend):
    print("\nLearning Performance:\n")
    episodes = []
    for i in range(len(total_rewards)):
        episodes.append(i * learning_num + 1)

    plt.figure(num=1)
    fig, ax = plt.subplots()
    plt.plot(episodes, total_rewards)
    plt.title('performance')
    plt.legend(legend)
    plt.xlabel("Episodes")
    plt.ylabel("total rewards")
    plt.show()


ray.shutdown()
ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=500000000, object_store_memory=5000000000)

# Set the Env name and action space for CartPole
ENV_NAME = 'CartPole_distributed'

# Set result saveing folder
result_folder = ENV_NAME + "_distributed"
result_file = ENV_NAME + "/results.txt"
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)
torch.set_num_threads(12)
env = CartPoleEnv()

hyperparams_CartPole = {
    'epsilon_decay_steps': 100000,
    'final_epsilon': 0.1,
    'batch_size': 32,
    'update_steps': 10,
    'memory_size': 2000,
    'beta': 0.99,
    'model_replace_episodes': 10,
    'learning_rate': 0.0003,
    'use_target_model': True
}

cw_num = 2
do_test = True

start_time = time.time()
distributed_agent = distributed_DQN_agent(env, hyperparams_CartPole, cw_num=cw_num, ew_num=4, do_test=do_test)
total_rewards = distributed_agent.learn_and_evaluate()
run_time = time.time() - start_time
print("Learning time:\n")
print(run_time)

plot_result(total_rewards, 100, ["distributed DQN"])
