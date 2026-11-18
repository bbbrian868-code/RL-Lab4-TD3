import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.replay_buffer import ReplayMemory
from abc import ABC, abstractmethod


class GaussianNoise:
	def __init__(self, dim, mu=None, std=None):
		self.mu = mu if mu else np.zeros(dim)
		self.std = np.ones(dim) * std if std else np.ones(dim) * .1
	
	def reset(self):
		pass

	def generate(self):
		return np.random.normal(self.mu, self.std)

class OUNoiseGenerator:
    def __init__(self, mean, std_dev, theta=0.3, dt=5e-2):
        self.theta = theta
        self.dt = dt
        self.mean = mean
        self.std_dev = std_dev

        self.x = None

        self.reset()

    def reset(self):
        self.x = np.zeros_like(self.mean.shape)

    def generate(self):
        self.x = (self.x
                  + self.theta * (self.mean - self.x) * self.dt
                  + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))

        return self.x

class TD3BaseAgent(ABC):
	def __init__(self, config):
		self.gpu = config["gpu"]
		self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
		self.total_time_step = 0
		self.training_steps = int(config["training_steps"])
		self.batch_size = int(config["batch_size"])
		self.warmup_steps = config["warmup_steps"]
		self.total_episode = config["total_episode"]
		self.eval_interval = config["eval_interval"]
		self.eval_episode = config["eval_episode"]
		self.gamma = config["gamma"]
		self.tau = config["tau"]
		self.update_freq = config["update_freq"]
		self.expl_noise = config.get("expl_noise", 0.1)
		self.seed = config.get("seed")
		# fixed tracks, so evaluation scores are comparable between runs
		self.eval_seeds = [10000 + i for i in range(self.eval_episode)]
		self.best_score = -np.inf
		# debug counters and the values of the latest update
		self.critic_update_count = 0
		self.actor_update_count = 0
		self.target_update_count = 0
		self.last_update_info = {}

		self.replay_buffer = ReplayMemory(int(config["replay_buffer_capacity"]))
		self.writer = SummaryWriter(config["logdir"])

	@abstractmethod
	def decide_agent_actions(self, state, sigma=0.0):
		### TODO ###
		# based on the behavior (actor) network and exploration noise
		
		return NotImplementedError
		
	
	def update(self):
		# update the behavior networks
		self.update_behavior_network()
		# update the target networks
		if self.total_time_step % self.update_freq == 0:
			self.update_target_network(self.target_actor_net, self.actor_net, self.tau)
			self.update_target_network(self.target_critic_net1, self.critic_net1, self.tau)
			self.update_target_network(self.target_critic_net2, self.critic_net2, self.tau)
			self.target_update_count += 1

	@abstractmethod
	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		
		### TODO ###
		# calculate the loss and update the behavior network

		return NotImplementedError
		

	@staticmethod
	def update_target_network(target_net, net, tau):
		# update target network by "soft" copying from behavior network
		for target, behavior in zip(target_net.parameters(), net.parameters()):
			target.data.copy_((1 - tau) * target.data + tau * behavior.data)
	
	def train(self):
		self.env.action_space.seed(self.seed)
		for episode in range(self.total_episode):
			total_reward = 0
			total_shaped_reward = 0
			state, infos = self.env.reset(seed=self.seed if episode == 0 else None)
			self.noise.reset()
			for t in range(10000):
				if self.total_time_step < self.warmup_steps:
					action = self.env.action_space.sample()
				else:
					# exploration degree
					sigma = max(self.expl_noise*(1-episode/self.total_episode), 0.01)
					action = self.decide_agent_actions(state, sigma=sigma)

				next_state, reward, terminates, truncates, info = self.env.step(action)
				self.replay_buffer.append(state, action, [reward/10], next_state, [int(terminates)])
				if self.total_time_step >= self.warmup_steps:
					self.update()

				self.total_time_step += 1
				# Train/Episode Reward always uses the original environment reward
				total_reward += info["original_reward"]
				total_shaped_reward += reward
				state = next_state
				if terminates or truncates:
					self.writer.add_scalar('Train/Episode Reward', total_reward, self.total_time_step)
					self.writer.add_scalar('Train/Shaped Reward', total_shaped_reward, self.total_time_step)
					print(
						'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}'
						.format(self.total_time_step, episode+1, t, total_reward))

				if self.total_time_step % self.eval_interval == 0:
					self.evaluate_and_save()

				if terminates or truncates or self.total_time_step >= self.training_steps:
					break

			if self.total_time_step >= self.training_steps:
				break

		print(f"Updates: critic={self.critic_update_count}, actor={self.actor_update_count}, target={self.target_update_count}")
		self.writer.close()

	def evaluate_and_save(self):
		# save the latest and the best model checkpoints
		avg_score = self.evaluate()
		self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)
		self.writer.flush()
		self.save(os.path.join(self.writer.log_dir, "latest.pth"))
		# keep every evaluated model, so a checkpoint can be chosen later on more tracks
		self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth"))
		if avg_score > self.best_score:
			self.best_score = avg_score
			self.save(os.path.join(self.writer.log_dir, "best.pth"))
			print(f"New best model at step {self.total_time_step}: {avg_score:.2f}")

	def evaluate(self, seeds=None):
		print("==============================================")
		print("Evaluating...")
		all_rewards = []
		seeds = self.eval_seeds if seeds is None else seeds
		for episode, seed in enumerate(seeds):
			total_reward = 0
			state, infos = self.test_env.reset(seed=seed)
			for t in range(10000):
				action = self.decide_agent_actions(state)
				next_state, reward, terminates, truncates, _ = self.test_env.step(action)
				total_reward += reward
				state = next_state
				if terminates or truncates:
					print(
						'Episode: {}\tLength: {:3d}\tTotal reward: {:.2f}'
						.format(episode+1, t, total_reward))
					all_rewards.append(total_reward)
					break

		avg = sum(all_rewards) / len(seeds)
		print(f"average score: {avg}")
		print("==============================================")
		return avg
	
	# save model
	def save(self, save_path):
		torch.save(
				{
					'actor': self.actor_net.state_dict(),
					'critic1': self.critic_net1.state_dict(),
					'critic2': self.critic_net2.state_dict(),
				}, save_path)

	# load model
	def load(self, load_path):
		checkpoint = torch.load(load_path, map_location=self.device)
		self.actor_net.load_state_dict(checkpoint['actor'])
		self.critic_net1.load_state_dict(checkpoint['critic1'])
		self.critic_net2.load_state_dict(checkpoint['critic2'])

	# load model weights and evaluate
	def load_and_evaluate(self, load_path):
		self.load(load_path)
		self.evaluate()

