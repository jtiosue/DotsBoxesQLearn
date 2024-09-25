import game
from typing import Tuple
from collections import deque
import torch
from torch import nn
import numpy as np
import random


WINNER_REWARD = 2
TIE_REWARD = 1
SQUARE_COMPLETED_REWARD = 0
LOSER_REWARD = 0


class Net(nn.Module):
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        player: int,
        hidden_layers: int = 3,
        hidden_dim: int = None,
        activation=nn.ReLU,
    ) -> None:
        super().__init__()

        self.args = num_rows, num_cols, player, hidden_layers, hidden_dim, activation
        self.player = player

        num_squares = num_rows * num_cols
        # see game.Board
        self.num_edges = num_rows * (num_cols + 1) + num_cols * (num_rows + 1)
        input_dim = self.num_edges + num_squares // 2
        if hidden_dim is None:
            hidden_dim = input_dim
        output_dim = self.num_edges
        if not hidden_layers:
            layers = [nn.Linear(input_dim, output_dim)]
        else:
            layers = [nn.Linear(input_dim, hidden_dim), activation()]
            for _ in range(hidden_layers):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation())
            layers.append(nn.Linear(hidden_dim, output_dim))

        layers.append(nn.ReLU())
        # layers.append(activation())

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        z = x
        for l in self.layers:
            z = l(z)
        # enforce that the move is valid
        return (1 - x[: self.num_edges]) * z

    def board_to_input(self, board: game.Board) -> torch.Tensor:
        squares = (
            torch.from_numpy(board.squares.flatten() == 1)
            .type(torch.float32)
            .sort(descending=True)[0]
        )
        squares = squares[: len(squares) // 2]
        grid = torch.from_numpy(board.grid).type(torch.float32)
        return torch.concatenate((grid, squares))

    def game_player(self, board: game.Board, exploration: float = 0.0) -> game.Move:
        # get all squares that the first player has claimed
        if random.random() <= exploration:
            move_index = random.randint(0, self.num_edges - 1)
            while not board.is_valid(board.index_to_move(move_index)):
                move_index = random.randint(0, self.num_edges - 1)
        else:
            with torch.no_grad():
                output = self(self.board_to_input(board))
            if torch.all(output == 0):  # so possibly invalid move
                return self.game_player(board, 1)
            move_index = torch.argmax(output)

        return board.index_to_move(move_index)

    def copy(self) -> "Net":
        net = Net(*self.args)
        net.load_state_dict(self.state_dict())
        return net

    def save(self, filename: str) -> None:
        torch.save(self.state_dict(), f"models/{filename}.pt")

    def load(self, filename):
        self.load_state_dict(torch.load(f"models/{filename}.pt"))


######### Adapated from https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_dql.py


# Define memory for Experience Replay
class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class Training:
    # Hyperparameters (adjustable)
    learning_rate_a = 0.05  # learning rate (alpha)
    discount_factor_g = 1  # discount rate (gamma)
    network_sync_rate = 20  # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 1000  # size of replay memory
    mini_batch_size = 100  # size training data set sampled from the replay memory

    hidden_layers: int = 8
    hidden_dim: int = None
    activation = nn.Tanh

    # Neural Network
    loss_fn = nn.MSELoss()
    # loss_fn = nn.BCELoss()

    def train(
        self,
        num_rows: int,
        num_cols: int,
        epochs: int = 1000,
        verbose=True,
        model1: Net = None,
        model2: Net = None,
        train_model1: bool = True,
        train_model2: bool = True,
    ) -> Tuple[Net, Net]:

        policy1 = model1 or Net(
            num_rows, num_cols, 1, self.hidden_layers, self.hidden_dim, self.activation
        )
        policy2 = model2 or Net(
            num_rows, num_cols, 2, self.hidden_layers, self.hidden_dim, self.activation
        )

        if not train_model1 and not train_model2:
            return policy1, policy2

        policies = {1: policy1, 2: policy2}
        train_models = {1: train_model1, 2: train_model2}
        targets = {1: policy1.copy(), 2: policy2.copy()}

        optimizers = {
            # 1: torch.optim.Adam(policy1.parameters(), lr=self.learning_rate_a),
            # 2: torch.optim.Adam(policy2.parameters(), lr=self.learning_rate_a),
            1: (
                torch.optim.SGD(
                    policy1.parameters(), lr=self.learning_rate_a, momentum=0.9
                )
                if train_model1
                else None
            ),
            2: (
                torch.optim.SGD(
                    policy2.parameters(), lr=self.learning_rate_a, momentum=0.9
                )
                if train_model2
                else None
            ),
        }
        memories = {
            1: ReplayMemory(self.replay_memory_size),
            2: ReplayMemory(self.replay_memory_size),
        }

        exploration = 1  # 1 = 100% random actions

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_epoch = {1: np.zeros(epochs), 2: np.zeros(epochs)}

        # List to keep track of exploration decay
        exploration_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count = 0

        for i in range(epochs):

            if verbose and epochs // 100 and not i % (epochs // 100):
                print(f"Starting epoch {i} out of {epochs}")

            playing, winner = True, 0
            cont_game = game.Game(
                num_rows,
                num_cols,
                lambda b: policy1.game_player(b, exploration if train_model1 else 0.15),
                lambda b: policy2.game_player(b, exploration if train_model2 else 0.15),
            )

            while playing:

                current_state = cont_game.board.copy()
                turn = cont_game.turn
                move = cont_game.play_step(verbose=False)
                if not move.valid:
                    print(move.row, move.col, move.vert)
                    print(current_state.move_to_index(move))
                    print(current_state)
                    with torch.no_grad():
                        output = policies[turn](
                            policies[turn].board_to_input(current_state)
                        )
                    print(output)
                    raise ValueError("Move not valid!")
                new_board = cont_game.board.copy()

                winner = cont_game.board.is_game_over()
                game_over = bool(winner)
                square_completed = cont_game.turn == turn

                if game_over:
                    playing = False
                    # If it is not a tie (winner != 3), then from game.Board.is_game_over, the winner must have
                    # # made the last move and completed a square
                    if winner != 3 and winner != turn:
                        raise ValueError("Something went wrong!!")

                    # reward = TIE_REWARD if winner == 3 else WINNER_REWARD
                    if turn == 1:
                        reward = 0 if winner == 3 else WINNER_REWARD
                    else:
                        reward = TIE_REWARD if winner == 3 else WINNER_REWARD
                else:
                    reward = SQUARE_COMPLETED_REWARD if square_completed else 0

                # Save experience into memory
                memories[turn].append(
                    (
                        current_state,
                        move,
                        new_board,
                        reward,
                        square_completed,
                        game_over,
                    )
                )

                # Increment step counter
                step_count += 1

                # Keep track of the rewards collected per episode.
                if reward:
                    rewards_per_epoch[turn][i] = reward

            # Check if enough experience has been collected and if at least 1 reward has been collected
            for player in (x for x in (1, 2) if train_models[x]):
                if (
                    len(memories[player]) > self.mini_batch_size
                    and np.sum(rewards_per_epoch[player]) > 0
                ):
                    # if verbose: print("Starting optimizer for player", player)
                    mini_batch = memories[player].sample(self.mini_batch_size)
                    self.optimize(
                        optimizers[player],
                        mini_batch,
                        policies[player],
                        targets[player],
                        targets[1 if player == 2 else 2],
                    )

                    # Copy policy network to target network after a certain number of steps
                    if step_count > self.network_sync_rate:
                        targets[player] = policies[player].copy()
                        step_count = 0

            # Decay exploration
            exploration = max(exploration - 1 / epochs, 0)
            exploration_history.append(exploration)

        return policy1, policy2

    # Optimize policy network
    def optimize(self, optimizer, mini_batch, policy, target, other_target):

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, square_completed, game_over in mini_batch:

            if game_over:
                # When in a terminated state, target q value should be set to the reward.
                targetq = torch.FloatTensor([reward])
            else:
                # square completed, so it's still the same player's turn
                # Calculate target q value
                if square_completed:
                    with torch.no_grad():
                        targetq = torch.FloatTensor(
                            reward
                            + self.discount_factor_g
                            * target(target.board_to_input(new_state)).max()
                        )
                else:
                    new_new_state = new_state.copy()
                    completed, winner = True, 0
                    while completed and not winner:
                        other_best_action = other_target.game_player(new_new_state)
                        completed = new_new_state.player_add_edge(
                            other_best_action, other_target.player
                        )
                        winner = new_new_state.is_game_over()

                    if not winner:
                        with torch.no_grad():
                            targetq = torch.FloatTensor(
                                reward
                                + self.discount_factor_g
                                * target(target.board_to_input(new_new_state)).max()
                            )
                    else:
                        # game is over by other person's moves
                        if winner == 3:
                            rew = TIE_REWARD
                        elif winner == target.player:
                            rew = WINNER_REWARD
                        else:
                            rew = LOSER_REWARD
                        targetq = torch.FloatTensor(
                            [reward + self.discount_factor_g * rew]
                        )

            # Get the current set of Q values
            current_q = policy(policy.board_to_input(state))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target(policy.board_to_input(state))
            # Adjust the specific action to the target that was just calculated
            target_q[state.move_to_index(action)] = targetq
            target_q_list.append(target_q)

        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":

    num_rows, num_cols = 2, 2

    model1, model2 = Training().train(num_rows, num_cols, epochs=1000)
    model1.save(f"{num_rows}{num_cols}_player1")
    model2.save(f"{num_rows}{num_cols}_player2")
    # print(
    #     game.Game(
    #         num_rows,
    #         num_cols,
    #         player1=model1.game_player,
    #         player2=model2.game_player,
    #     ).play(False)
    # )

    # for _ in range(5):
    #     model1, model2 = Training().train(
    #         num_rows,
    #         num_cols,
    #         epochs=1000,
    #         verbose=False,
    #         model1=model1,
    #         model2=model2,
    #         train_model1=True,
    #         train_model2=False,
    #     )
    #     model1.save(f"{num_rows}{num_cols}_player1")
    #     model2.save(f"{num_rows}{num_cols}_player2")
    #     winner = game.Game(
    #         num_rows,
    #         num_cols,
    #         player1=model1.game_player,
    #         player2=model2.game_player,
    #     ).play(False)
    #     print(f"winner = {winner} (Should be 1)")

    #     model1, model2 = Training().train(
    #         num_rows,
    #         num_cols,
    #         epochs=1000,
    #         verbose=False,
    #         model1=model1,
    #         model2=model2,
    #         train_model1=False,
    #         train_model2=True,
    #     )
    #     model1.save(f"{num_rows}{num_cols}_player1")
    #     model2.save(f"{num_rows}{num_cols}_player2")
    #     winner = game.Game(
    #         num_rows,
    #         num_cols,
    #         player1=model1.game_player,
    #         player2=model2.game_player,
    #     ).play(False)
    #     print(f"winner = {winner} (Should be 1, 2 or 3)")

    # model1, model2 = Training().train(
    #     num_rows, num_cols, epochs=1000, model1=model1, model2=model2
    # )
    # model1.save(f"{num_rows}{num_cols}_player1")
    # model2.save(f"{num_rows}{num_cols}_player2")

    model1, model2 = Training().train(
        num_rows, num_cols, train_model1=False, train_model2=False
    )
    model1.load(f"{num_rows}{num_cols}_player1")
    model2.load(f"{num_rows}{num_cols}_player2")

    from vanilla_ql import QTable

    van_model1 = QTable(num_rows, num_cols, 1)
    van_model1.load_qmatrix(f"{num_rows}{num_cols}_player1")
    van_model2 = QTable(num_rows, num_cols, 2)
    van_model2.load_qmatrix(f"{num_rows}{num_cols}_player2")

    for _ in range(10):
        winner = game.Game(
            num_rows,
            num_cols,
            player1=model1.game_player,
            # player1=game.user_player,
            # player1=lambda b: van_model1.game_player(b, 0.0),
            # player2=lambda b: model2.game_player(b, 0.0),
            # player2=game.user_player,
            player2=lambda b: van_model2.game_player(b, 1),
        ).play(False)
        print(winner)
