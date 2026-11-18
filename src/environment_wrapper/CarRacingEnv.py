import argparse
from collections import deque
import itertools
import random
import time
import cv2

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class RewardShapingWrapper(gym.Wrapper):
	"""Custom training reward, enabled with --reward_shaping. It is never used for evaluation.
	no_reward_patience=None disables the early stop, reward_low=None disables the clip."""
	def __init__(self, env, position_fn, grass_penalty=0.5, no_reward_patience=100, reward_low=-1.0, crop_pixels=480):
		super().__init__(env)
		self.position_fn = position_fn
		self.grass_penalty = grass_penalty
		self.no_reward_patience = no_reward_patience
		self.reward_low = reward_low
		self.crop_pixels = crop_pixels  # size of the crop in check_car_position (24 x 20)
		self.no_reward_steps = 0

	def clip_reward(self, reward):
		# limit large penalties such as the -100 for leaving the map or the road
		if self.reward_low is None:
			return reward
		return max(reward, self.reward_low)

	def reset(self, **kwargs):
		self.no_reward_steps = 0
		return self.env.reset(**kwargs)

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		info["raw_reward"] = reward

		# off-track penalty, proportional to the grass pixels around the car
		_, grass_pixel_count = self.position_fn(obs)
		shaped_reward = reward - self.grass_penalty * grass_pixel_count / self.crop_pixels
		shaped_reward = self.clip_reward(shaped_reward)

		# early stop: this is a truncation, so the critic still bootstraps from the last state
		self.no_reward_steps = 0 if reward > 0 else self.no_reward_steps + 1
		if self.no_reward_patience is not None and self.no_reward_steps >= self.no_reward_patience:
			truncated = True

		return obs, shaped_reward, terminated, truncated, info

class CarRacingEnvironment:
	def __init__(self, N_frame=4, test=False, render_mode="default", record_video_dir=None, reward_shaping=False):
		self.test = test
		# reward shaping only affects training
		self.reward_shaping = False if test else reward_shaping
		# keep the sample code's behavior when render_mode is not given
		if render_mode == "default":
			render_mode = "human" if self.test else None
		if record_video_dir is not None:
			render_mode = "rgb_array"
		self.env = gym.make('CarRacing-v2', render_mode=render_mode)
		if record_video_dir is not None:
			self.env = RecordVideo(self.env, record_video_dir, episode_trigger=lambda episode: True, disable_logger=True)
		if self.reward_shaping == "grass":
			# grass penalty only: the -100 off-road penalty is kept, no clip and no early stop
			self.env = RewardShapingWrapper(self.env, self.check_car_position, no_reward_patience=None, reward_low=None)
		elif self.reward_shaping:
			# grass penalty + early stop + clip (run H)
			self.env = RewardShapingWrapper(self.env, self.check_car_position)
		self.action_space = self.env.action_space
		self.observation_space = self.env.observation_space
		self.ep_len = 0
		self.frames = deque(maxlen=N_frame)
	
	def check_car_position(self, obs):
		# cut the image to get the part where the car is
		part_image = obs[60:84, 40:60, :]

		road_color_lower = np.array([90, 90, 90], dtype=np.uint8)
		road_color_upper = np.array([120, 120, 120], dtype=np.uint8)
		grass_color_lower = np.array([90, 180, 90], dtype=np.uint8)
		grass_color_upper = np.array([120, 255, 120], dtype=np.uint8)
		road_mask = cv2.inRange(part_image, road_color_lower, road_color_upper)
		grass_mask = cv2.inRange(part_image, grass_color_lower, grass_color_upper)
		# count the number of pixels in the road and grass
		road_pixel_count = cv2.countNonZero(road_mask)
		grass_pixel_count = cv2.countNonZero(grass_mask)

		# save image for debugging
		# filename = "images/image" + str(self.ep_len) + ".jpg"
		# cv2.imwrite(filename, part_image)

		return road_pixel_count, grass_pixel_count

	def step(self, action):
		obs, reward, terminates, truncates, info = self.env.step(action)
		original_reward = info.get("raw_reward", reward)
		original_terminates = terminates
		self.ep_len += 1
		road_pixel_count, grass_pixel_count = self.check_car_position(obs)
		info["road_pixel_count"] = road_pixel_count
		info["grass_pixel_count"] = grass_pixel_count
		# the agent always logs this reward as Train/Episode Reward
		info["original_reward"] = original_reward

		# my reward shaping strategy, you can try your own
		if road_pixel_count < 10:
			terminates = True
			reward = -100
			if self.reward_shaping:
				reward = self.env.clip_reward(reward)

		# convert to grayscale
		obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY) # 96x96

		# save image for debugging
		# filename = "images/image" + str(self.ep_len) + ".jpg"
		# cv2.imwrite(filename, obs)

		# frame stacking
		self.frames.append(obs)
		obs = np.stack(self.frames, axis=0)

		if self.test:
			# enable this line to recover the original reward
			reward = original_reward
			# enable this line to recover the original terminates signal, disable this to accerlate evaluation
			terminates = original_terminates

		return obs, reward, terminates, truncates, info
	
	def reset(self, seed=None):
		obs, info = self.env.reset(seed=seed)
		self.ep_len = 0
		obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY) # 96x96

		# frame stacking
		for _ in range(self.frames.maxlen):
			self.frames.append(obs)
		obs = np.stack(self.frames, axis=0)

		return obs, info
	
	def render(self):
		self.env.render()
	
	def close(self):
		self.env.close()

if __name__ == '__main__':
	env = CarRacingEnvironment(test=True)
	obs, info = env.reset()
	done = False
	total_reward = 0
	total_length = 0
	t = 0
	while not done:
		t += 1
		action = env.action_space.sample()
		action[2] = 0.0
		obs, reward, terminates, truncates, info = env.step(action)
		print(f'{t}: road_pixel_count: {info["road_pixel_count"]}, grass_pixel_count: {info["grass_pixel_count"]}, reward: {reward}')
		total_reward += reward
		total_length += 1
		env.render()
		if terminates or truncates:
			done = True

	print("Total reward: ", total_reward)
	print("Total length: ", total_length)
	env.close()
