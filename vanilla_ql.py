import game
import random, collections

WINNER_REWARD = 100
TIE_REWARD = 10
SQUARE_COMPLETED_REWARD = 1
LOSER_REWARD = -100


class QTable:

    def __init__(self, num_rows: int, num_cols: int) -> None:
        self.dim = len(game.Board(num_rows, num_cols).grid)
        self.Q = collections.defaultdict(float)
        self.args = num_rows, num_cols

    def copy(self) -> "QTable":
        q = QTable(*self.args)
        q.Q = self.Q.copy()
        return q

    def load_qmatrix(self, filename: str) -> None:
        with open(f"models/{filename}.van") as f:
            Q = eval(f.read())
        self.Q = collections.defaultdict(float)
        for k, v in Q.items():
            self.Q[k] = v

    def write_qmatrix(self, filename: str) -> None:
        with open(f"models/{filename}.van", "w") as f:
            f.write(str({k: v for k, v in self.Q.items() if v}))

    def get_optimal_action(self, board: game.Board, player: int) -> int:
        # returns an index
        h = board.hash(player)
        mvs = list(range(self.dim))
        random.shuffle(mvs)
        move = max(mvs, key=lambda x: self.Q[(h, x)])
        while board.grid[move]:  # invalid move
            self.Q[(h, move)] -= 100
            move = max(mvs, key=lambda x: self.Q[(h, x)])
        return move

    def bellman(
        self,
        turn: int,
        square_completed: bool,
        board: game.Board,
        action: int,
        new_board: game.Board,
        reward: float,
        lr: float,
        discount: float,
        game_over: bool,
    ) -> None:

        board_hash = board.hash(turn)
        if game_over:
            self.Q[(board_hash, action)] = reward
            return

        if square_completed:  # square completed, so it's still the same QTable's turn
            new_board_hash = new_board.hash(turn)
            self.Q[(board_hash, action)] = (1 - lr) * self.Q[
                (board_hash, action)
            ] + lr * (
                reward
                + discount * max(self.Q[(new_board_hash, a)] for a in range(self.dim))
            )

        else:
            # it is the other person's turn. So we take the Q-matrix for the other player,
            # figure out their optimal move, and then figure out our future reward based on
            # the assumption that the other player is playing optimally

            new_board_hash = new_board.hash(1 if turn == 2 else 2)
            self.Q[(board_hash, action)] = (1 - lr) * self.Q[
                (board_hash, action)
            ] + lr * (
                reward
                - discount * max(self.Q[(new_board_hash, a)] for a in range(self.dim))
            )

    def game_player(
        self, board: game.Board, player: int, exploration: float = 0.0
    ) -> game.Move:
        if random.random() <= exploration:
            move = random.randint(0, self.dim - 1)
            while not board.is_valid(board.index_to_move(move)):
                self.Q[(board.hash(player), move)] -= 100
                move = random.randint(0, self.dim - 1)
        else:
            move = self.get_optimal_action(board, player)
        return board.index_to_move(move)

    def game_player1(self, board: game.Board, exploration: float = 0.0) -> game.Move:
        return self.game_player(board, 1, exploration)

    def game_player2(self, board: game.Board, exploration: float = 0.0) -> game.Move:
        return self.game_player(board, 2, exploration)


def train_qtables(
    num_rows: int,
    num_cols: int,
    epochs: int = 1000,
    initial_exploration: float = 1,
    initial_lr=0.1,
    discount=1,
    verbose=True,
    model: QTable = None,
    train_model: bool = True,
) -> QTable:

    model = model or QTable(num_rows, num_cols)
    epochs = int(epochs)

    if not train_model:
        return model

    lr, exploration = initial_lr, initial_exploration

    for epoch in range(epochs):
        if epochs // 100 and not epoch % (epochs // 100):
            if verbose:
                print(f"Starting epoch {epoch} out of {epochs}")
            exploration /= 1.05

        lr = initial_lr / (1 + epoch // (1 + epochs // 100))

        cont_game = game.Game(
            num_rows,
            num_cols,
            lambda b: model.game_player1(b, exploration),
            lambda b: model.game_player2(b, exploration),
        )

        playing, winner = True, 0
        while playing:
            current_board = cont_game.board.copy()
            turn = cont_game.turn
            move = cont_game.play_step(verbose=False)
            new_board = cont_game.board.copy()
            # if not move.valid:
            #   do something
            move_index = cont_game.board.move_to_index(move)
            if winner := cont_game.board.is_game_over():
                playing = False
            else:  # if we are training this model
                square_completed = cont_game.turn == turn
                model.bellman(
                    turn,
                    square_completed=square_completed,
                    board=current_board,
                    action=move_index,
                    new_board=new_board,
                    reward=SQUARE_COMPLETED_REWARD if square_completed else -1,
                    lr=lr,
                    discount=discount,
                    game_over=False,
                )

        # If it is not a tie (winner != 3), then from game.Board.is_game_over, the winner must have
        # # made the last move and completed a square
        if winner != 3 and winner != turn:
            raise ValueError("Something went wrong!!")

        if winner == 3:
            rew = TIE_REWARD if turn == 2 else -TIE_REWARD
        else:
            rew = WINNER_REWARD

        model.bellman(
            turn,
            square_completed=cont_game.turn == turn,
            board=current_board,
            action=move_index,
            new_board=new_board,
            reward=rew,
            lr=lr,
            discount=discount,
            game_over=True,
        )

    return model


if __name__ == "__main__":
    num_rows, num_cols = 2, 2

    # uncomment these lines to train the model
    # model = train_qtables(num_rows, num_cols, epochs=1e6, verbose=True)
    # model.write_qmatrix(f"{num_rows}{num_cols}")

    # these lines load the model
    model = QTable(num_rows, num_cols)
    model.load_qmatrix(f"{num_rows}{num_cols}")

    # to play against the model, uncomment the following:
    # game.Game(
    #     num_rows,
    #     num_cols,
    #     player1=model.game_player1,
    #     # player1=game.user_player  # this is you
    #     # player2=model.game_player2,
    #     player2=game.user_player,  # this is you
    # ).play()

    # note that player 1 can always force a win
    # let's see if the model always wins as player 1
    winners = {1: 0, 2: 0, 3: 0}
    for _ in range(10000):
        winner = game.Game(
            num_rows,
            num_cols,
            player1=model.game_player1,
            player2=lambda b: model.game_player2(b, exploration=0.2),
        ).play(False)
        winners[winner] += 1
    print(winners)

    # If we add some randomness to player 1, then the model as player 2
    # should now be able to win sometimes
    winners = {1: 0, 2: 0, 3: 0}
    for _ in range(10000):
        winner = game.Game(
            num_rows,
            num_cols,
            player1=lambda b: model.game_player1(b, exploration=0.2),
            player2=model.game_player2,
        ).play(False)
        winners[winner] += 1
    print(winners)
