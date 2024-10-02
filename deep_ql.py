import game
from collections import deque
import torch
from torch import nn
import random


WINNER_REWARD = 20
TIE_REWARD = 3
SQUARE_COMPLETED_REWARD = 1
LOSER_REWARD = -20


class Net(nn.Module):
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        hidden_layers: int = 3,
        hidden_dim: int = None,
        activation=nn.ReLU,
    ) -> None:
        super().__init__()

        self.args = num_rows, num_cols, hidden_layers, hidden_dim, activation

        # see game.Board
        self.num_edges = num_rows * (num_cols + 1) + num_cols * (num_rows + 1)
        input_dim = self.num_edges + 1
        if hidden_dim is None:
            hidden_dim = 2 * input_dim
        output_dim = self.num_edges
        if not hidden_layers:
            layers = [nn.Linear(input_dim, output_dim)]
        else:
            layers = [nn.Linear(input_dim, hidden_dim), activation()]
            for _ in range(hidden_layers):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation())
            layers.append(nn.Linear(hidden_dim, output_dim))

        # layers.append(nn.ReLU())
        # layers.append(activation())

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        z = x
        for l in self.layers:
            z = l(z)
        # enforce that the move is valid
        return z + LOSER_REWARD * x[: self.num_edges]

    def board_to_input(self, board: game.Board, player: int) -> torch.Tensor:
        grid = torch.from_numpy(board.grid).type(torch.float32)
        score = board.score()
        # sq = torch.FloatTensor([score[player - 1]])
        sq = torch.FloatTensor([score[player - 1] - score[player % 2]])
        return torch.concatenate((grid, sq))

    def game_player(
        self, board: game.Board, player: int, exploration: float = 0.0
    ) -> game.Move:
        # get all squares that the first player has claimed
        if random.random() <= exploration:
            move_index = random.randint(0, self.num_edges - 1)
            while not board.is_valid(board.index_to_move(move_index)):
                move_index = random.randint(0, self.num_edges - 1)
        else:
            with torch.no_grad():
                output = self(self.board_to_input(board, player))
            move_index = torch.argmax(output)

        return board.index_to_move(move_index)

    def game_player1(self, board: game.Board, exploration: float = 0.0) -> game.Move:
        return self.game_player(board, 1, exploration)

    def game_player2(self, board: game.Board, exploration: float = 0.0) -> game.Move:
        return self.game_player(board, 2, exploration)

    def copy(self) -> "Net":
        net = Net(*self.args)
        net.load_state_dict(self.state_dict())
        return net

    def save(self, filename: str) -> None:
        torch.save(self.state_dict(), f"models/{filename}.pt")

    def load(self, filename):
        self.load_state_dict(torch.load(f"models/{filename}.pt"))


######### Adapated from https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_dql.py
######## https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


# Define memory for Experience Replay
class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
        # avoid catostrophic forgetting
        # https://datascience.stackexchange.com/questions/56053/why-could-my-ddqn-get-significantly-worse-after-beating-the-game-repeatedly
        self.initial_memory = []
        self.maxlen = maxlen

    def append(self, transition):
        self.memory.append(transition)
        if len(self.initial_memory) < self.maxlen:
            self.initial_memory.append(transition)

    def sample(self, sample_size):
        return random.sample(list(self.memory) + self.initial_memory, sample_size)

    def __len__(self):
        return len(self.memory)


class Training:
    # Hyperparameters (adjustable)
    learning_rate_a = 0.01  # learning rate (alpha)
    discount_factor_g = 0.99  # discount rate (gamma)
    network_sync_rate = 20  # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 1000  # size of replay memory
    mini_batch_size = 128  # size training data set sampled from the replay memory

    tau = 0.01

    hidden_layers: int = 7
    hidden_dim: int = None
    activation = nn.ReLU

    # Neural Network
    # loss_fn = nn.MSELoss()
    loss_fn = nn.SmoothL1Loss()
    # loss_fn = nn.BCELoss()

    loss = []

    def train(
        self,
        num_rows: int,
        num_cols: int,
        epochs: int = 1000,
        verbose=True,
        model: Net = None,
        train_model: bool = True,
    ) -> Net:

        policy = model or Net(
            num_rows, num_cols, self.hidden_layers, self.hidden_dim, self.activation
        )

        if not train_model:
            return policy

        epochs = int(epochs)

        target = policy.copy()

        optimizer = torch.optim.AdamW(
            policy.parameters(), lr=self.learning_rate_a, amsgrad=True
        )
        memory = ReplayMemory(self.replay_memory_size)

        exploration = 1  # 1 = 100% random actions

        # Track number of steps taken. Used for syncing policy => target network.
        step_count = 0

        for i in range(epochs):

            if verbose and epochs // 100 and not i % (epochs // 100):
                print(f"Starting epoch {i} out of {epochs}")

            playing, winner = True, 0
            cont_game = game.Game(
                num_rows,
                num_cols,
                lambda b: policy.game_player1(b, exploration),
                lambda b: policy.game_player2(b, exploration),
            )
            policy.eval()
            target.eval()

            while playing:

                current_state = cont_game.board.copy()
                turn = cont_game.turn
                move = cont_game.play_step(verbose=False)
                if not move.valid:
                    # print(move.row, move.col, move.vert)
                    # print(current_state.move_to_index(move))
                    # print(current_state)
                    # with torch.no_grad():
                    #     output = policies[turn](
                    #         policies[turn].board_to_input(current_state)
                    #     )
                    # print(output)
                    # raise ValueError("Move not valid!")
                    playing = False
                    game_over = True
                    winner = 1 if turn == 2 else 2
                    reward = 2 * LOSER_REWARD
                    continue

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
                        reward = -TIE_REWARD if winner == 3 else WINNER_REWARD
                    else:
                        reward = TIE_REWARD if winner == 3 else WINNER_REWARD
                else:
                    reward = (
                        SQUARE_COMPLETED_REWARD
                        if square_completed
                        else -SQUARE_COMPLETED_REWARD
                    )

                # Save experience into memory
                memory.append(
                    (
                        current_state,
                        turn,
                        move,
                        new_board,
                        reward,
                        square_completed,
                        game_over,
                    )
                )

                # Increment step counter
                step_count += 1

            # Check if enough experience has been collected and if at least 1 reward has been collected
            if len(memory) > self.mini_batch_size:
                # if verbose: print("Starting optimizer for player", player)
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(
                    optimizer,
                    mini_batch,
                    policy,
                    target,
                )

                # Copy policy network to target network after a certain number of steps
                # if step_count > self.network_sync_rate:
                #     target = policy.copy()
                #     step_count = 0

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_state_dict = target.state_dict()
                policy_state_dict = policy.state_dict()
                for key in policy_state_dict:
                    target_state_dict[key] = policy_state_dict[
                        key
                    ] * self.tau + target_state_dict[key] * (1 - self.tau)
                target.load_state_dict(target_state_dict)

            # Decay exploration
            exploration = max(exploration - 1 / epochs, 0)

        print(self.loss)

        return policy

    # Optimize policy network
    def optimize(self, optimizer, mini_batch, policy, target):

        current_q_list = []
        target_q_list = []

        policy.train()

        for (
            state,
            turn,
            action,
            new_state,
            reward,
            square_completed,
            game_over,
        ) in mini_batch:

            if game_over:
                # When in a terminated state, target q value should be set to the reward.
                targetq = torch.FloatTensor([reward])
            # square completed, so it's still the same player's turn
            # Calculate target q value
            elif square_completed:
                with torch.no_grad():
                    targetq = torch.FloatTensor(
                        reward
                        + self.discount_factor_g
                        * target(target.board_to_input(new_state, turn)).max()
                        # * policy(target.board_to_input(new_state, turn)).max()
                    )
            else:
                with torch.no_grad():
                    targetq = torch.FloatTensor(
                        reward
                        - self.discount_factor_g
                        * target(
                            target.board_to_input(new_state, 1 if turn == 2 else 2)
                        ).max()
                        # * policy(
                        #     target.board_to_input(new_state, 1 if turn == 2 else 2)
                        # ).max()
                    )

            # Get the current set of Q values
            current_q = policy(policy.board_to_input(state, turn))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target(target.board_to_input(state, turn))
            # Adjust the specific action to the target that was just calculated
            target_q[state.move_to_index(action)] = targetq
            target_q_list.append(target_q)

        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
        self.loss.append(float(loss))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy.parameters(), 100)
        optimizer.step()


if __name__ == "__main__":

    num_rows, num_cols = 2, 2

    # model = Training().train(num_rows, num_cols, epochs=5 * 1e4)
    # model.save(f"{num_rows}{num_cols}")

    model = Training().train(num_rows, num_cols, train_model=False)
    model.load(f"{num_rows}{num_cols}")

    model.eval()

    from vanilla_ql import QTable

    van_model = QTable(num_rows, num_cols)
    van_model.load_qmatrix(f"{num_rows}{num_cols}")

    winners = {1: 0, 2: 0, 3: 0}
    for _ in range(1000):
        winner = game.Game(
            num_rows,
            num_cols,
            # player1=model.game_player1,
            # player1=game.user_player,
            player1=lambda b: van_model.game_player1(b, 1),
            player2=lambda b: model.game_player2(b, 0),
            # player2=game.user_player,
            # player2=lambda b: van_model.game_player2(b, 0),
        ).play(False)
        winners[winner] += 1
    print(winners)
