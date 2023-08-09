import numpy as np
import torch
import os
import torch.nn.functional as F
from ResNet import ResNet
from ResNet import MCTSNode
from connect_four_gymnasium import ConnectFourEnv

NUM_BLOCKS = 8
NUM_CHANNELS = 128
LEARNING_RATE = 0.001
MCTS_ITERATIONS = 100
BATCH_SIZE = 64

if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))


def get_device():
    """Returns the device to be used for computation (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AlphaFour:
    def __init__(self, model_path=None,iteration=MCTS_ITERATIONS):
        """Initializes the AlphaFour class with a ResNet model and an optimizer.

        Args:
            model_path (str, optional): Path to the pre-trained model. Defaults to None.
        """
        self.iteration = iteration

        self.model = ResNet(NUM_BLOCKS, NUM_CHANNELS)
        self.model.to(get_device())

        if model_path is not None and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=get_device()))
        elif model_path is not None:
            exit('Model not found' + str(model_path))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.result_cache = {}
        self.leaf_cache = {}

    def reset_cache(self):
        """Resets the result_cache and leaf_cache dictionaries."""
        self.result_cache = {}
        self.leaf_cache = {}

    def save_model(self, model_path):
        """Saves the current model to the specified path.

        Args:
            model_path (str): Path to save the model.
        """
        torch.save(self.model.state_dict(), model_path)

    def obs_to_tensor(self, states):
        """Converts a list of states to a tensor.

        Args:
            states (list): List of states.

        Returns:
            torch.Tensor: Tensor representation of the states.
        """
        states = np.stack(states)
        encoded_states = np.stack((states == 1, states == -1)).swapaxes(0, 1)
        return torch.tensor(encoded_states, dtype=torch.float32).to(get_device())

    def transform_to_unique_id(self, board):
        """Converts a board state to a unique identifier.

        Args:
            board (np.array): Board state.

        Returns:
            int: Unique identifier for the board state.
        """
        board_str = ""
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                board_str += str(board[i, j] + 1)
        return int(board_str, base=3)

    def mcts_parallel(self, envs, use_model=True):
        """Performs Monte Carlo Tree Search (MCTS) in parallel for multiple environments.

        Args:
            envs (list): List of environments.
            use_model (bool, optional): Whether to use the model for MCTS. Defaults to True.

        Returns:
            list: List of policies and values for each environment.
        """

        #We don't want to calcul twice the same board
        #so we begin with an distinct operation
        #then we filter the board already in the cache
        unique_envs_dict = {self.transform_to_unique_id(env.board): env for env in envs}
        filtered_unique_envs = list({unique_id: env for unique_id, env in unique_envs_dict.items() if unique_id not in self.result_cache}.values())
        cached = len(envs) - len(filtered_unique_envs)

        #We calcul the mcts for the board not in the cache
        if len(filtered_unique_envs) > 0:
            policies, values = self.mcts_parallel_no_result_cache(filtered_unique_envs, use_model)
            for env, policy, value in zip(filtered_unique_envs, policies, values):
                unique_id = self.transform_to_unique_id(env.board)
                self.result_cache[unique_id] = policy, value

        #We add all the respone from the cache
        cached = [self.result_cache[self.transform_to_unique_id(env.board)] for env in envs]
        policies = [policy for policy, value in cached]
        values = [value for policy, value in cached]

        return policies, values

    def set_node_in_leaf_cache(self, root: MCTSNode, recursive: bool = True):
        """Sets a node in the leaf_cache dictionary.

        Args:
            root (MCTSNode): The root node to be set in the cache.
            recursive (bool, optional): Whether to set the node recursively. Defaults to True.
        """
        unique_identifier = self.transform_to_unique_id(root.env.board)
        if unique_identifier in self.leaf_cache:
            cached_node = self.leaf_cache[unique_identifier]
            if cached_node.visit_count < root.visit_count:
                self.leaf_cache[unique_identifier] = root
        else:
            self.leaf_cache[unique_identifier] = root

        if recursive:
            for child in root.children:
                self.set_node_in_leaf_cache(child, False)
                for childchild in child.children:
                    self.set_node_in_leaf_cache(childchild, False)

    def create_node_with_leaf_cache(self, env: ConnectFourEnv):
        """Creates a new MCTSNode with the given environment, using the leaf_cache if available.

        Args:
            env (ConnectFourEnv): The environment for the new node.

        Returns:
            MCTSNode: The created node.
        """
        unique_identifier = self.transform_to_unique_id(env.board)
        if unique_identifier in self.leaf_cache:
            return self.leaf_cache[unique_identifier].deepcopy()
        else:
            return MCTSNode(env)

    def train(self, examples):
        """Trains the model using the given examples.

        Args:
            examples (list): List of training examples.

        Returns:
            float: Mean loss for the training examples.
        """
        self.reset_cache()

        unique_examples = {}

        for state, policy, value in examples:
            unique_id = self.transform_to_unique_id(state)
            if unique_id not in unique_examples:
                unique_examples[unique_id] = (state, policy, value)

        unique_examples_list = list(unique_examples.values())

        self.model.train()
        np.random.shuffle(unique_examples_list)
        losses = []
        for i in range(0, len(unique_examples_list), BATCH_SIZE):
            states, policies, values = zip(*unique_examples_list[i: i + BATCH_SIZE])
            policies = torch.tensor(np.array(policies), dtype=torch.float32).to(get_device())
            values = torch.tensor(np.array(values), dtype=torch.float32).unsqueeze(1).to(get_device())
            pred_policies, pred_values = self.model(self.obs_to_tensor(states))
            loss = F.cross_entropy(pred_policies, policies) + F.mse_loss(pred_values, values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return np.mean(losses)

    def mcts_parallel_no_result_cache(self, envs, use_model: bool):
        """Performs Monte Carlo Tree Search (MCTS) in parallel for multiple environments without using result_cache.

        Args:
            envs (list): List of environments.
            use_model (bool): Whether to use the model for MCTS.

        Returns:
            list: List of policies and values for each environment.
        """
        self.model.eval()
        roots = [self.create_node_with_leaf_cache(env) for env in envs]

        for i in range(self.iteration):
            leafs = []

            for root in roots:

                node = root
                while len(node.children) > 0:
                    node = node.select()

                if node.is_finish == True:
                    node.backprop(node.win)
                elif node.is_finish == False:
                    leafs.append(node)
                else:
                    result, is_finish = node.env.is_finish()
                    node.is_finish = is_finish
                    node.win = result
                    if is_finish:
                        node.backprop(result)
                    else:
                        leafs.append(node)

            if len(leafs) > 0:
                leaf_envs = [leaf.env for leaf in leafs]
                leaf_board = [env.board for env in leaf_envs]

                if use_model:
                    with torch.no_grad():
                        policies, values = self.model(self.obs_to_tensor(leaf_board))

                    policies = torch.softmax(policies, 1).cpu().numpy()
                    values = values.squeeze(1).cpu().numpy()
                else:
                    policies = np.ones((len(leafs), 7)) / 7
                    values = np.zeros(len(leafs))

                for j in range(len(leafs)):
                    policies[j][[i for i in range(len(policies[j])) if i not in leaf_envs[j].get_valid_actions()]] = 0.0
                    policies[j] /= np.sum(policies[j])
                    leafs[j].expand(policies[j])
                    leafs[j].backprop(values[j])
        policies = []
        values = []
        for root in roots:
            action_size = root.env.COLUMNS_COUNT
            policy = np.zeros(action_size)
            for child in root.children:
                policy[child.prev_action] = child.visit_count
            policy /= np.sum(policy)

            policies.append(policy)
            children_with_best_policy = [child for child in root.children if child.visit_count == max([child.visit_count for child in root.children])]

            values.append([root.value_sum / root.visit_count, children_with_best_policy])

        for root in roots:
            self.set_node_in_leaf_cache(root)

        return policies, values