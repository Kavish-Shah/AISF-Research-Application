import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
import os

class ForwardPriorityWrapper(gym.Wrapper):
    """
    A custom wrapper that incentivizes the agent to move forward 
    by adding a bonus to the reward based on its X-velocity.
    """
    def __init__(self, env, forward_weight=10.0):
        super().__init__(env)
        self.forward_weight = forward_weight

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        
        forward_velocity = obs[2]
        
        reward += forward_velocity * self.forward_weight
        
        return obs, reward, terminated, truncated, info

models_dir = "models/PPO"
log_dir = "logs"
video_dir = "videos"

if not os.path.exists(models_dir): os.makedirs(models_dir)
if not os.path.exists(log_dir): os.makedirs(log_dir)
if not os.path.exists(video_dir): os.makedirs(video_dir)

def make_env():
    env = gym.make('BipedalWalker-v3', render_mode='rgb_array')
    
    env = Monitor(env, log_dir)
    
    env = ForwardPriorityWrapper(env, forward_weight=10.0)
    
    return env

env = DummyVecEnv([make_env])

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

print("Training Model with Forward Priority...")

model.learn(total_timesteps=100000) 
print("Training finished.")

model.save(f"{models_dir}/forward_priority_model")

print("Recording video...")


video_env = DummyVecEnv([make_env])

video_env = VecVideoRecorder(video_env, video_dir,
                           record_video_trigger=lambda x: x == 0,
                           video_length=1500,
                           name_prefix="forward-agent")

obs = video_env.reset()
for _ in range(1500 + 1):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = video_env.step(action)

video_env.close()
print(f"Video saved to {video_dir} folder!")