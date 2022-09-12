import gym
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback,CheckpointCallback
import time

# ---------------------------------------------------------------------------- #
#                                Argument Parser                               #
# ---------------------------------------------------------------------------- #
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "mode",
    choices=['train','test'],
    help="select training or testing"
)
parser.add_argument(
    '-l',
    '--load',
    help="load model from given path"
)
args = parser.parse_args()

# ---------------------------------------------------------------------------- #
#                          using GPU if it's available                         #
# ---------------------------------------------------------------------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"running on '{device}'")


# ---------------------------------------------------------------------------- #
#                             creating environment                             #
# ---------------------------------------------------------------------------- #
game = "LunarLander-v2"
# game = "CartPole-v1"
env = gym.make(game)

TIME_STEPS = 200000

# ---------------------------------------------------------------------------- #
#                                Train function                                #
# ---------------------------------------------------------------------------- #
def train(start_checkpoint=None):
    """
    Train agent using SB3 implemented DQN on the environment
    """

    if start_checkpoint:
        model = DQN.load(start_checkpoint)
    
    else:
        model = DQN(
            'MlpPolicy',
            env,
            device=device,
            tensorboard_log='./tb_logs',
            verbose=1,
        )

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/',
                                                    name_prefix='dqn')

    callbacks = []
    callbacks.append(checkpoint_callback)

    kwargs = {}
    kwargs["callback"] = callbacks

    model.learn(
            total_timesteps=TIME_STEPS,
            tb_log_name=game + str(time.time()),
            log_interval=50,
            **kwargs
        )
    
    model.save(game + "_final_model")



# ---------------------------------------------------------------------------- #
#                                 Test Function                                #
# ---------------------------------------------------------------------------- #
def test(checkpoint):
    model = DQN.load(checkpoint)

    count = 0
    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        env.render()

        time.sleep(0.02)

        if done:
            count+=1
            if count >= 2:
                env.close()
                break
            else:
                obs = env.reset()







if __name__ == "__main__":
    if args.mode == "train":
        if args.load:
            train(args.load)
        else:
            train()
    else:
        if args.load:
            print(f"loading {args.load}")
            test(args.load)
        else:
            test(f"./{game}_final_model.zip")
    