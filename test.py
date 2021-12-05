# test PPO implementation in simple environments

from src.Algorithms.PPO import PPO
import shutil, os
import gym
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

if __name__=="__main__":
    env = gym.make("CartPole-v1")

    if os.path.exists("logging/test"):
        shutil.rmtree("logging/test")

    model = PPO(
        state_dimension=4,
        action_dimension=2,
        lr_actor=1e-4,
        lr_critic=3e-4,
        train=True,
        num_epochs=80,
        discount=0.99,
        eps_clip=0.2,
        max_grad_norm=0.2,
        batch_size=64
    )
    
    max_eq_len = 400
    time_step = 0
    update_timestep = 4000
    render_timestep = int(500)

    writer = SummaryWriter("logging/test")

    for episode in range(5000):
        done = False
        episode_reward = 0
        state = env.reset().tolist()

        for t in range(1, max_eq_len + 1):
            action = model.action([state])
            state, reward, done, _ = env.step(action)
            state = state.tolist()
            model.update(reward, done)
            episode_reward += reward
            time_step += 1
            writer.add_scalar("Reward/Timestep", episode_reward, time_step)

            if time_step % update_timestep == 0:
                print("PPO Training")
                model.train(writer)
                print("PPO Finished")

            # if episode % render_timestep == 0:
            #     env.render()

            if done:
                break
        
        writer.add_scalar("Reward/Episodic", episode_reward, episode)

    writer.close()