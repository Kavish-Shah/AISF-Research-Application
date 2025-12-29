import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor  
import os

# 1. Setup Directories
models_dir = "models/PPO"
log_dir = "logs"
video_dir = "videos"

if not os.path.exists(models_dir): os.makedirs(models_dir)
if not os.path.exists(log_dir): os.makedirs(log_dir)
if not os.path.exists(video_dir): os.makedirs(video_dir)

# 2. Create the Environment (Now with Monitor!)
def make_env():
    env = gym.make('BipedalWalker-v3', render_mode='rgb_array')
    # This wrapper records the episode rewards (score) and length
    env = Monitor(env, log_dir) 
    return env

# Create the vectorized environment
env = DummyVecEnv([make_env])

# 3. Create the Model (Baseline - Default Settings)
# We stick to the defaults here to show "standard" performance
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

# 4. Train the Model
print("Training Baseline Model...")
model.learn(total_timesteps=100000)
print("Training finished.")

# 5. Save the Model
model.save(f"{models_dir}/baseline_model")

# 6. Evaluate and Record Video
print("Recording video...")
# We recreate the env for video recording
video_env = DummyVecEnv([make_env])

video_env = VecVideoRecorder(video_env, video_dir,
                           record_video_trigger=lambda x: x == 0,
                           video_length=1500,
                           name_prefix="baseline-agent")

obs = video_env.reset()
for _ in range(1500 + 1):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = video_env.step(action)

video_env.close()
print(f"Video saved to {video_dir} folder!")