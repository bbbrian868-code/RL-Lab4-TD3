import torch
import torch.nn as nn
import numpy as np
from base_agent import TD3BaseAgent
from models.CarRacing_model import ActorNetSimple, CriticNetSimple
from environment_wrapper.CarRacingEnv import CarRacingEnvironment
import random
from base_agent import OUNoiseGenerator, GaussianNoise

class CarRacingTD3Agent(TD3BaseAgent):
	def __init__(self, config):
		super(CarRacingTD3Agent, self).__init__(config)
		# initialize environment
		self.env = CarRacingEnvironment(N_frame=4, test=False, reward_shaping=config.get("reward_shaping", False))
		self.test_env = CarRacingEnvironment(N_frame=4, test=True, render_mode=config.get("render_mode"), record_video_dir=config.get("record_video_dir"))

		# ablation switches (see main.py for the CLI flags)
		self.single_q = config.get("single_q", False)
		self.target_smoothing = config.get("target_smoothing", True)
		# not given by the lab handout: TD3 paper defaults
		self.target_noise = config.get("target_noise", 0.2)
		self.noise_clip = config.get("noise_clip", 0.5)
		# action bounds: steering [-1, 1], gas [0, 1], brake [0, 1]
		self.action_low = torch.as_tensor(self.env.action_space.low, dtype=torch.float, device=self.device)
		self.action_high = torch.as_tensor(self.env.action_space.high, dtype=torch.float, device=self.device)

		# behavior network
		self.actor_net = ActorNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.critic_net1 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.critic_net2 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.actor_net.to(self.device)
		self.critic_net1.to(self.device)
		self.critic_net2.to(self.device)
		# target network
		self.target_actor_net = ActorNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.target_critic_net1 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.target_critic_net2 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.target_actor_net.to(self.device)
		self.target_critic_net1.to(self.device)
		self.target_critic_net2.to(self.device)
		self.target_actor_net.load_state_dict(self.actor_net.state_dict())
		self.target_critic_net1.load_state_dict(self.critic_net1.state_dict())
		self.target_critic_net2.load_state_dict(self.critic_net2.state_dict())
		
		# set optimizer
		self.lra = config["lra"]
		self.lrc = config["lrc"]
		
		self.actor_opt = torch.optim.Adam(self.actor_net.parameters(), lr=self.lra)
		self.critic_opt1 = torch.optim.Adam(self.critic_net1.parameters(), lr=self.lrc)
		self.critic_opt2 = torch.optim.Adam(self.critic_net2.parameters(), lr=self.lrc)

		# choose Gaussian noise or OU noise

		# noise_mean = np.full(self.env.action_space.shape[0], 0.0, np.float32)
		# noise_std = np.full(self.env.action_space.shape[0], 1.0, np.float32)
		# self.noise = OUNoiseGenerator(noise_mean, noise_std)

		self.noise = GaussianNoise(self.env.action_space.shape[0], 0.0, 1.0)
		# per-dimension scale of the exploration noise: steering, gas, brake
		# the actor's brake output is only in [0, 2 * brake_rate], so --brake_noise_scale can shrink the brake noise
		self.expl_noise_scale = np.array([1.0, 1.0, config.get("brake_noise_scale", 1.0)])

	
	def decide_agent_actions(self, state, sigma=0.0, brake_rate=0.015):
		### TODO ###
		# based on the behavior (actor) network and exploration noise
		with torch.no_grad():
			state = torch.as_tensor(np.asarray(state), dtype=torch.float, device=self.device).unsqueeze(0) / 255.0
			action = self.actor_net(state, brake_rate).squeeze(0).cpu().numpy() + sigma * self.noise.generate() * self.expl_noise_scale

		# clip every dimension to its own valid range
		return np.clip(action, self.env.action_space.low, self.env.action_space.high).astype(np.float32)


	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		# states are stored as uint8 and normalized only after sampling
		state = state / 255.0
		next_state = next_state / 255.0
		### TODO ###
		### TD3 ###
		# 1. Clipped Double Q-Learning for Actor-Critic
		# 2. Delayed Policy Updates
		# 3. Target Policy Smoothing Regularization

		## Update Critic ##
		# critic loss
		q_value1 = self.critic_net1(state, action)
		if not self.single_q:
			q_value2 = self.critic_net2(state, action)
		with torch.no_grad():
			# select action a_next from target actor network and add noise for smoothing
			a_next = self.target_actor_net(next_state)
			if self.target_smoothing:
				noise = (torch.randn_like(a_next) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
				a_next = a_next + noise
			# the actor's gas output can reach 1.1, so clip even without smoothing noise
			a_next = a_next.clamp(self.action_low, self.action_high)

			q_next1 = self.target_critic_net1(next_state, a_next)
			if self.single_q:
				q_next = q_next1
			else:
				q_next2 = self.target_critic_net2(next_state, a_next)
				# select min q value from q_next1 and q_next2 (double Q learning)
				q_next = torch.min(q_next1, q_next2)
			# done only marks true terminal states, a time-limit truncation still bootstraps
			q_target = reward + self.gamma * (1 - done) * q_next

		# critic loss function
		criterion = nn.MSELoss()
		critic_loss1 = criterion(q_value1, q_target)

		# optimize critic
		# the critics have separate parameters and optimizers,
		# so this is equivalent to minimizing critic_loss1 + critic_loss2
		self.critic_net1.zero_grad()
		critic_loss1.backward()
		self.critic_opt1.step()

		if not self.single_q:
			critic_loss2 = criterion(q_value2, q_target)
			self.critic_net2.zero_grad()
			critic_loss2.backward()
			self.critic_opt2.step()
		self.critic_update_count += 1

		self.last_update_info.update({
			"critic_loss1": critic_loss1.item(),
			"critic_loss2": None if self.single_q else critic_loss2.item(),
			"q_min": q_value1.min().item(),
			"q_max": q_value1.max().item(),
			"q_target_min": q_target.min().item(),
			"q_target_max": q_target.max().item(),
		})

		## Delayed Actor(Policy) Updates ##
		if self.total_time_step % self.update_freq == 0:
			## update actor ##
			# actor loss
			# select action a from behavior actor network (a is different from sample transition's action)
			# get Q from behavior critic network, mean Q value -> objective function
			# maximize (objective function) = minimize -1 * (objective function)
			action = self.actor_net(state)
			actor_loss = -1 * self.critic_net1(state, action).mean()
			# optimize actor
			self.actor_net.zero_grad()
			actor_loss.backward()
			self.actor_opt.step()
			self.actor_update_count += 1
			self.last_update_info["actor_loss"] = actor_loss.item()

