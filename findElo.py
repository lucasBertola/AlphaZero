import torch
import numpy as np
from connect_four_gymnasium import ConnectFourEnv
from connect_four_gymnasium.players.Player import Player
from connect_four_gymnasium.tools import EloLeaderboard
from alphaFour import AlphaFour

VALUE_PLAYER_GAMES = 300
POLICY_DETER_GAMES = 300
POLICY_GAMES = 300
MCTS_GAMES = 100

class ValuePlayer(Player):
    def __init__(self, ia: AlphaFour, name="Value", deterministic=True):
        self.ia = ia
        self.name = name
        self.deterministic = deterministic
        self.elo = None

    def play_single(self, observation):
        env = ConnectFourEnv()
        env.reset()
        env.board = observation
        valid_actions = env.get_valid_actions()
        min_value = float('inf')
        best_action = None
        envsactions = [(env.clone(), action) for action in valid_actions]
        for env, action in envsactions:
            env.play_action(action)
            env.switch_player()

        allenv = [env.board for env, action in envsactions]

        with torch.no_grad():
            _, values = self.ia.model(self.ia.obs_to_tensor(allenv))

        for action, value in zip(valid_actions, values):
            if value < min_value:
                min_value = value
                best_action = action

        return best_action

    def play(self, observations):
        if isinstance(observations, list):
            best_actions = []
            for obs in observations:
                best_action = self.play_single(obs)
                best_actions.append(best_action)
            return np.array(best_actions)
        else:
            return np.array([self.play_single(observations)])

    def get_name(self):
        return self.name

    def is_deterministic(self):
        return self.deterministic

    def getElo(self):
        return self.elo

    def set_elo(self, elo):
        self.elo = elo


class PolicyPlayer(Player):
    def __init__(self, ia: AlphaFour, name="Policy", deterministic=True):
        self.ia = ia
        self.name = name
        self.deterministic = deterministic
        self.elo = None

    def play(self, observations):
        self.ia.model.eval()
        with torch.no_grad():
            pred_policies, pred_values = self.ia.model(self.ia.obs_to_tensor(observations))
        pred_policies = pred_policies.detach().cpu().numpy()

        actions = []
        for policy in pred_policies:
            if self.deterministic:
                actions.append(np.argmax(policy))
            else:
                action_choices = np.arange(len(policy))
                probabilities = torch.softmax(torch.tensor(policy), dim=0).numpy()
                chosen_action = np.random.choice(action_choices, p=probabilities)
                actions.append(chosen_action)

        return np.array(actions)

    def get_name(self):
        return self.name

    def is_deterministic(self):
        return self.deterministic

    def getElo(self):
        return self.elo

    def set_elo(self, elo):
        self.elo = elo


class MCTSPlayer(Player):
    def __init__(self, ia: AlphaFour, name="MCTS", deterministic=True):
        self.ia = ia
        self.name = name
        self.deterministic = deterministic
        self.elo = None
        self.use_model = False

    def play(self, observations):
        if isinstance(observations, list):
            envs = [ConnectFourEnv() for i in range(len(observations))]
            for i, env in enumerate(envs):
                env.reset()
                env.board = observations[i]
            pred_policies, values = self.ia.mcts_parallel(envs, self.use_model)
            best_actions = np.argmax(pred_policies, axis=1)
            return best_actions
        else:
            env = ConnectFourEnv()
            env.reset()
            env.board = observations
            pred_policies, values = self.ia.mcts_parallel([env], self.use_model)
            return np.argmax(pred_policies, axis=1)

    def get_name(self):
        return self.name

    def is_deterministic(self):
        return self.deterministic

    def set_elo(self, elo):
        self.elo = elo

    def getElo(self):
        return self.elo


class EloFinder:
    def __init__(self, ia: AlphaFour, name="MCTS"):
        self.policy_player_deter = PolicyPlayer(ia, 'Policy deter', deterministic=True)
        self.policy_player = PolicyPlayer(ia, 'Policy', deterministic=False)
        self.value_player = ValuePlayer(ia, 'ValuePlayer', deterministic=False)
        self.player_mcts = MCTSPlayer(ia, 'MCTS')
        self.ia = ia

    def find_and_display_elo(self, generation):
        self.ia.model.eval()
        print(f'Finding elo for generation {generation}')

        elo_value = EloLeaderboard().get_elo(self.value_player, VALUE_PLAYER_GAMES)
        self.value_player.set_elo(elo_value)
        print(f'Elo of {self.value_player.get_name()}: {round(elo_value)}')

        elo_policy_deter = EloLeaderboard().get_elo(self.policy_player_deter, POLICY_DETER_GAMES, True)
        self.policy_player_deter.set_elo(elo_policy_deter)
        print(f'Elo of {self.policy_player_deter.get_name()}: {round(elo_policy_deter)}')

        elo_policy = EloLeaderboard().get_elo(self.policy_player, POLICY_GAMES, True)
        self.policy_player.set_elo(elo_policy)
        print(f'Elo of {self.policy_player.get_name()}: {round(elo_policy)}')

        self.player_mcts.use_model = generation != 0
        elo_mcts = EloLeaderboard().get_elo(self.player_mcts, MCTS_GAMES, True)
        self.player_mcts.set_elo(elo_mcts)
        print(f'Elo of {self.player_mcts.get_name()}: {round(elo_mcts)}')