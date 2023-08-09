import numpy as np
import os
import time

from connect_four_gymnasium import ConnectFourEnv
from connect_four_gymnasium.players import ConsolePlayer
from alphaFour import AlphaFour


class MCTSPlayer:
    def __init__(self, ai: AlphaFour, name="MCTS", deterministic=True):
        self.ai = ai
        self.name = name
        self.deterministic = deterministic

    def play_single(self, observation):
        env = ConnectFourEnv()
        env.reset()
        env.board = observation
        predicted_policies, _ = self.ai.mcts_parallel([env])
        return np.argmax(predicted_policies, axis=1)

    def play(self, observations):
        if isinstance(observations, list):
            envs = [ConnectFourEnv() for _ in range(len(observations))]
            for i, env in enumerate(envs):
                env.reset()
                env.board = observations[i]
            predicted_policies, values = self.ai.mcts_parallel(envs)
        else:
            env = ConnectFourEnv()
            env.reset()
            env.board = observations
            predicted_policies, values = self.ai.mcts_parallel([env])

        print(f'IA: I think I have a {round(((values[0][0] + 1) / 2) * 100)}% chance of winning')
        return np.argmax(predicted_policies, axis=1)

    def get_name(self):
        return self.name

    def is_deterministic(self):
        return self.deterministic

    def get_elo(self):
        return None


def find_latest_model():
    for i in reversed(range(500)):
        path = f'models/model_{i}.pt'
        if os.path.exists(path):
            return path, i

    return None, None


def main():
    latest_model_path, generation = find_latest_model()

    if latest_model_path is not None:
        print(f'Loading model generation {generation}')
        ai_instance = AlphaFour(latest_model_path)
    else:
        print('Error: No model found')
        exit()

    mcts_player = MCTSPlayer(ai_instance, 'MCTS')
    human_player = ConsolePlayer()
    env = ConnectFourEnv(opponent=mcts_player, render_mode="human")

    observation, _ = env.reset()
    for _ in range(5000):
        action = human_player.play(observation)
        observation, rewards, done, truncated, _ = env.step(action)
        env.render()
        if truncated or done:
            observation, _ = env.reset()


if __name__ == '__main__':
    main()