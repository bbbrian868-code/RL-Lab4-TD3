import argparse
import os
import tempfile

from td3_agent_CarRacing import CarRacingTD3Agent


def parse_args():
	parser = argparse.ArgumentParser(description="Evaluate a TD3 checkpoint on CarRacing-v2")
	parser.add_argument("--checkpoint", type=str, required=True, help="path to a .pth checkpoint")
	parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="one track per seed")
	parser.add_argument("--render", action="store_true", help="show the game window")
	parser.add_argument("--record_video", type=str, default=None, metavar="DIR", help="save one video per episode")
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()

	config = {
		"gpu": True,
		"training_steps": 0,
		"gamma": 0.99,
		"tau": 0.005,
		"batch_size": 32,
		"warmup_steps": 0,
		"total_episode": 0,
		"lra": 4.5e-5,
		"lrc": 4.5e-5,
		"replay_buffer_capacity": 1,
		# the agent always creates a SummaryWriter, testing does not use it
		"logdir": os.path.join(tempfile.gettempdir(), "td3_lab4_test_log"),
		"update_freq": 2,
		"eval_interval": 1,
		"eval_episode": len(args.seeds),
		"render_mode": "human" if args.render else None,
		"record_video_dir": args.record_video,
	}
	agent = CarRacingTD3Agent(config)
	agent.load(args.checkpoint)
	agent.evaluate(seeds=args.seeds)
	# closing the environment writes out the last video
	agent.test_env.close()
	agent.env.close()
	agent.writer.close()
