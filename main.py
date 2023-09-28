import gym
from ppo import PPO


env = gym.make('Pendulum-v1')
hyperparameters = {
    'n_steps': 128,
    'n_timesteps_per_batch': 4800,
    'max_timesteps_per_episode': 1600,
    'n_updates_per_iteration': 5,
    'gamma': 0.99,
    'lr': 2.5e-4,
    'clip_range': 0.2,
    'anneal_lr': False,
    'cuda': True
}
model = PPO(env)
model.learn(100000)

obs = env.reset()
done = False
while not done:
    env.render()
    action = model.get_action(obs)
    obs, reward, done, _ = env.step(action)
    #time.sleep(0.01)