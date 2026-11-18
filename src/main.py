import argparse
import os
import random

import cv2
import numpy as np
import torch

from td3_agent_CarRacing import CarRacingTD3Agent

DEFAULT_POLICY_DELAY = 2
DEFAULT_EXPL_NOISE = 0.1
DEFAULT_BUFFER_SIZE = 5000
DEFAULT_BRAKE_NOISE_SCALE = 1.0
# <project>/runs, outside src/ so that logs and weights are never submitted
RUNS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs"))


def parse_args():
	parser = argparse.ArgumentParser(description="TD3 on CarRacing-v2")
	parser.add_argument("--single_q", action="store_true", help="train only Q1, the target does not take a min")
	parser.add_argument("--no_target_smoothing", action="store_true", help="no noise on the target action")
	parser.add_argument("--policy_delay", type=int, default=DEFAULT_POLICY_DELAY, help="actor/target update interval")
	parser.add_argument("--expl_noise", type=float, default=DEFAULT_EXPL_NOISE, help="exploration noise sigma")
	parser.add_argument("--brake_noise_scale", type=float, default=DEFAULT_BRAKE_NOISE_SCALE,
		help="multiplier of the exploration noise on the brake dimension only")
	parser.add_argument("--buffer_size", type=int, default=DEFAULT_BUFFER_SIZE, help="replay buffer capacity")
	parser.add_argument("--reward_shaping", nargs="?", const="v1", default=None, choices=["v1", "grass"],
		help="custom training reward: v1 (default) = grass penalty + early stop + clip, grass = grass penalty only")
	parser.add_argument("--total_steps", type=int, default=1000000, help="total environment steps")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--run_name", type=str, default=None, help="log/checkpoint name, generated if omitted")
	parser.add_argument("--eval_interval", type=int, default=20000, help="environment steps between evaluations")
	parser.add_argument("--eval_episode", type=int, default=3, help="episodes per evaluation")
	return parser.parse_args()


def make_run_name(args):
	parts = ["td3"]
	if args.single_q:
		parts.append("singleQ")
	if args.no_target_smoothing:
		parts.append("noSmooth")
	if args.policy_delay != DEFAULT_POLICY_DELAY:
		parts.append(f"delay{args.policy_delay}")
	if args.expl_noise != DEFAULT_EXPL_NOISE:
		parts.append(f"expl{args.expl_noise}")
	if args.brake_noise_scale != DEFAULT_BRAKE_NOISE_SCALE:
		parts.append(f"brake{args.brake_noise_scale}")
	if args.buffer_size != DEFAULT_BUFFER_SIZE:
		parts.append(f"buf{args.buffer_size}")
	if args.reward_shaping == "v1":
		parts.append("shaped")
	elif args.reward_shaping == "grass":
		parts.append("shapedGrass")
	parts.append(f"seed{args.seed}")
	return "_".join(parts)


if __name__ == '__main__':
	args = parse_args()
	run_name = args.run_name or make_run_name(args)

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	# several trainings run in parallel and the bottleneck is the CPU
	torch.set_num_threads(2)
	cv2.setNumThreads(1)

	# my hyperparameters, you can change it as you like
	config = {
		"gpu": True,
		"training_steps": args.total_steps,
		"gamma": 0.99,
		"tau": 0.005,
		"batch_size": 32,
		"warmup_steps": 1000,
		"total_episode": 100000,
		"lra": 4.5e-5,
		"lrc": 4.5e-5,
		"replay_buffer_capacity": args.buffer_size,
		"logdir": os.path.join(RUNS_DIR, run_name),
		"update_freq": args.policy_delay,
		"eval_interval": args.eval_interval,
		"eval_episode": args.eval_episode,
		"seed": args.seed,
		"single_q": args.single_q,
		"target_smoothing": not args.no_target_smoothing,
		"expl_noise": args.expl_noise,
		"brake_noise_scale": args.brake_noise_scale,
		"reward_shaping": args.reward_shaping,
		# not given by the lab handout: TD3 paper defaults
		"target_noise": 0.2,
		"noise_clip": 0.5,
	}
	print(f"Run: {run_name}\nLog directory: {config['logdir']}")
	agent = CarRacingTD3Agent(config)
	agent.train()
