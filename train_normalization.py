import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecNormalize # <--- IMPORT THIS
from stable_baselines3.common.monitor import Monitor  
import os

models_dir = "models/PPO_Improved"
log_dir = "logs"
video_dir = "videos"

if not os.path.exists(models_dir): os.makedirs(models_dir)
if not os.path.exists(log_dir): os.makedirs(log_dir)
if not os.path.exists(video_dir): os.makedirs(video_dir)

def make_env():
    env = gym.make('BipedalWalker-v3', render_mode='rgb_array')
    env = Monitor(env, log_dir) 
    return env

env = DummyVecEnv([make_env])

# Normalize Observations
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# Bigger Network
policy_kwargs = dict(net_arch=[400, 300])

model = PPO('MlpPolicy', env, verbose=1, 
            tensorboard_log=log_dir, 
            policy_kwargs=policy_kwargs,
            learning_rate=0.0003
            )

print("Training Improved Model (This takes longer)...")
model.learn(total_timesteps=300000, tb_log_name="03_solution_normalized") 
print("Training finished.")

model.save(f"{models_dir}/improved_model")
env.save(f"{models_dir}/vec_normalize.pkl") 

print("Recording video...")
video_env = DummyVecEnv([make_env])
video_env = VecNormalize.load(f"{models_dir}/vec_normalize.pkl", video_env) 
video_env.training = False
video_env.norm_reward = False

video_env = VecVideoRecorder(video_env, video_dir,
                           record_video_trigger=lambda x: x == 0,
                           video_length=1500,
                           name_prefix="improved-agent")

obs = video_env.reset()
for _ in range(1500 + 1):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = video_env.step(action)

video_env.close()
print("Experiment 3 Complete!")