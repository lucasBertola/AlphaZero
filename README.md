# Introduction: The Power of Alpha Zero

Alpha Zero is a groundbreaking artificial intelligence (AI) algorithm developed by DeepMind, a subsidiary of Alphabet Inc. It made headlines in 2017 when it defeated the world champion of Go, a complex board game with more possible positions than there are atoms in the universe. This remarkable achievement demonstrated the immense potential of AI, as Go is considered to be a much more challenging game than chess, which had already been conquered by AI in the 1990s.

The success of Alpha Zero lies in its ability to learn and improve its gameplay without any human intervention. Unlike traditional AI algorithms that rely on pre-programmed strategies and heuristics, Alpha Zero learns by playing against itself, generating better and better players with each generation. This self-learning approach allows the algorithm to discover novel strategies and tactics, ultimately achieving superhuman performance in a variety of board games.

## Alpha Zero Implementation

This repository presents a complete and efficient implementation of the Alpha Zero algorithm. While the original Alpha Zero was designed to play Go, our implementation is flexible and can be connected to any board game. In this version, we have chosen to connect it to Connect Four, a popular and relatively simpler game.

Our implementation showcases not only the learning process of Alpha Zero but also provides concrete results, including win rates, move quality, and other performance metrics. This allows users to witness the incredible capabilities of Alpha Zero and gain a deeper understanding of its inner workings.

## Learning Process of Alpha Zero

Alpha Zero learns by playing games against itself, without any human guidance or pre-programmed strategies. It starts with a randomly initialized neural network and improves its gameplay through a process called self-play. In each game, Alpha Zero explores the game tree using Monte Carlo Tree Search (MCTS) and updates its neural network based on the outcomes of the games.

With each generation, the algorithm becomes a better player, discovering new strategies and tactics along the way. The Elo rating, a measure of a player's skill level, is used to track the progress of Alpha Zero over time. As shown in the results below, the Elo rating of Alpha Zero increases with each generation, demonstrating its ability to learn and improve its gameplay autonomously.

In conclusion, Alpha Zero is an extraordinary AI algorithm that has revolutionized the field of artificial intelligence and board games. Its ability to learn and improve without human intervention has led to superhuman performance in various games, including the highly complex game of Go. Our implementation of Alpha Zero offers a practical and accessible way to explore this powerful algorithm and witness its incredible capabilities firsthand.

# The Two Key Components of Alpha Zero

Alpha Zero is a groundbreaking AI algorithm that has achieved superhuman performance in various board games. Its success lies in its ability to learn and improve its gameplay without any human intervention or pre-programmed strategies. To understand how Alpha Zero works, let's break it down into its two key components:

#### 1. A Powerful Neural Network

Imagine you have a neural network complex enough to learn to play at the same level as a player. The only requirement for learning to play at this level is to collect a large amount of data from that player and feed it to the neural network so that it can learn and adapt. This neural network serves as the foundation for Alpha Zero's learning process.

#### 2. A Self-Improving Algorithm: Monte Carlo Tree Search (MCTS)

Now, imagine you have an algorithm that takes the input from the neural network (the bot) and ensures that it becomes better than that bot. The algorithm in question is the MCTS. It uses the MCTS on its own model to become better, then plays a lot of games against itself to make its neural network learn to be as strong.

So imagine the MCTS gives you always 1 Elo point. It's easy to understand how it works:

1. Your network starts at 0 Elo (it doesn't know how to play).
2. Thanks to the MCTS, it will have 1 Elo.
3. The algorithm will play a lot of games against itself.
4. The neural network will learn from these games, so the neural network will have 1 Elo by definition.
5. Now, the MCTS will have 2 Elo by definition
6. The process is repeated, and the algorithm continues to improve.

This simple yet powerful approach allows Alpha Zero to learn and improve its gameplay autonomously, ultimately achieving superhuman performance in various board games.

# Play Alpha Zero Online

You can play against our Alpha Zero implementation using this [Google Colab notebook](https://colab.research.google.com/github/lucasBertola/AlphaZero/blob/main/Play_again_alpha_zero.ipynb). To get started, simply click on the link and follow the instructions provided in the notebook.

**Important:** Make sure to activate the GPU in the notebook to ensure optimal performance. To do this, go to `Runtime` > `Change runtime type` and select `GPU` under `Hardware accelerator`. Enjoy your game against Alpha Zero!

# Evaluating Alpha Zero's Elo Rating

To assess the performance of our Alpha Zero implementation, we have evaluated its Elo rating over time. The Elo rating system is a widely used method for calculating the relative skill levels of players in two-player games. In our case, we have used the [Connect-4-Gym-env-Reinforcement-learning library](https://github.com/lucasBertola/Connect-4-Gym-env-Reinforcement-learning) to evaluate the Elo ratings of Alpha Zero's MCTS, policy, and value components.

For reference, a good adult player who is familiar with the game typically has an Elo rating of around 2000.

The three graphs below illustrate the evolution of Elo ratings for Alpha Zero's MCTS, policy, and value components:

1. **Alpha Zero MCTS Elo:** This graph shows the evolution of the Elo rating for Alpha Zero's MCTS component. As the algorithm learns and improves its gameplay, the Elo rating of the MCTS increases, demonstrating the effectiveness of the self-improving algorithm.

![Alpha Zero MCTS Elo](https://github.com/lucasBertola/AlphaZero/blob/main/img/elo_alphazero_large.jpg?raw=true)

2. **Policy Elo:** This graph displays the Elo rating evolution of the policy component. To simulate a player that "plays the policy," we created a player who makes moves according to the probability of the best move given by the policy, in a weighted manner. As the algorithm learns, the Elo rating of the policy component also increases.

![Policy Elo](https://github.com/lucasBertola/AlphaZero/blob/main/img/elo_policy.jpg?raw=true)

3. **Value Elo:** This graph shows the Elo rating evolution of the value component. To simulate a player that "plays the value," we calculated the value for all possible moves and selected the move with the highest probability of winning. As the algorithm learns, the Elo rating of the value component increases as well.

![Value Elo](https://github.com/lucasBertola/AlphaZero/blob/main/img/elo_value.jpg?raw=true)

These graphs demonstrate the remarkable learning capabilities of Alpha Zero, as its Elo ratings for MCTS, policy, and value components all increase over time. This highlights the effectiveness of the self-improving algorithm and its ability to achieve superhuman performance in various board games.

# Contribute and Support Our Project

We hope you find our Alpha Zero implementation both informative and enjoyable. If you appreciate our work and would like to support us, please consider giving our repository a star on GitHub. This will help us reach a wider audience and encourage further development.

Additionally, we welcome any improvements or suggestions you may have. If you would like to contribute to our project, feel free to submit a pull request with your proposed changes. We are always eager to learn from the community and collaborate on enhancing our implementation.

Thank you for your interest in our Alpha Zero project, and we look forward to hearing your feedback and ideas for improvement!
