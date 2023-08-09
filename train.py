import numpy as np
import os
from connect_four_gymnasium import ConnectFourEnv
from alphaFour import AlphaFour
from findElo import EloFinder

GAMES_PER_GEN = 500
EPOCHS_PER_GEN = 10
NUM_GENS = 17
DIRICHLET_EPSILON = 0.10
DIRICHLET_ALPHA = 10


def self_play(ai: AlphaFour, generation: int):
    """
    Conduct self-play for the given AI model and generation.

    :param ai: The AI model to use for self-play.
    :param generation: The current generation of the AI model.
    :return: A list of all moves from the completed games.
    """
    all_moves_ended_games = []
    unfinished_games = [ConnectFourEnv() for _ in range(GAMES_PER_GEN)]
    [game.reset() for game in unfinished_games]
    moves_of_current_games = [[] for _ in range(GAMES_PER_GEN)]
    move = 0

    while unfinished_games:
        policies, values = ai.mcts_parallel(unfinished_games, generation > 1)
        finished_indices = []

        for i in reversed(range(len(unfinished_games))):
            moves_of_current_games[i].append([unfinished_games[i].board.copy(), policies[i], values[i][0]])
            noise = np.random.dirichlet(7 * [DIRICHLET_ALPHA], 1)
            policies_dirichlet = (1.0 - DIRICHLET_EPSILON) * policies[i] + DIRICHLET_EPSILON * noise
            action = np.random.choice(unfinished_games[i].COLUMNS_COUNT, p=policies_dirichlet[0])
            new_board, result, is_finished, is_truncated, _ = unfinished_games[i].step(action)

            if is_finished:
                # Add symmetries moves
                sym_moves = []
                for move_of_current_games in moves_of_current_games[i]:
                    sym_board = np.flip(move_of_current_games[0], axis=1)
                    sym_policy = np.flip(move_of_current_games[1])
                    sym_value = move_of_current_games[2]
                    sym_moves.append([sym_board, sym_policy, sym_value])
                all_moves_ended_games += moves_of_current_games[i]
                all_moves_ended_games += sym_moves
                finished_indices.append(i)

        unfinished_games = [game for i, game in enumerate(unfinished_games) if i not in finished_indices]
        moves_of_current_games = [moves for i, moves in enumerate(moves_of_current_games) if i not in finished_indices]
        move += 1

    return all_moves_ended_games


def find_latest_model():
    """
    Find the latest saved model.

    :return: The path to the latest saved model and its generation number.
    """
    for i in reversed(range(NUM_GENS)):
        path = f'models/model_{i}.pt'
        if os.path.exists(path):
            return path, i
    return None, None


def main():
    last_model_path, starting_generation = find_latest_model()

    if last_model_path is not None:
        print('Loading model generation ', starting_generation)
        ai = AlphaFour(last_model_path)
    else:
        ai = AlphaFour()
        ai.save_model('models/model_0.pt')
        starting_generation = 0

    elo_finder = EloFinder(ai)

    elo_finder.find_and_display_elo(starting_generation)

    for i in range(NUM_GENS):
        generation = starting_generation + i + 1
        print(f'generation: {generation}/{starting_generation + NUM_GENS}')
        examples = self_play(ai, generation)
        print('train..')
        for epoch in range(EPOCHS_PER_GEN):
            ai.train(examples)
        ai.save_model(f'models/model_{generation}.pt')
        elo_finder.find_and_display_elo(generation)


if __name__ == '__main__':
    main()