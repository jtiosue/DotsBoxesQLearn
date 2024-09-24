import game
from typing import Tuple
import random, collections

WINNER_REWARD = 100
TIE_REWARD = 3
SQUARE_COMPLETED_REWARD = 1
LOSER_REWARD = -5


class QTable:

    def __init__(self, num_rows: int, num_cols: int, player: int) -> None:
        self.dim = len(game.Board(num_rows, num_cols).grid)
        self.Q = collections.defaultdict(float)
        self.player = player  # player 1 or player 2

    def load_qmatrix(self, filename: str) -> None:
        with open(f"models/{filename}.van") as f:
            Q = eval(f.read())
        self.Q = collections.defaultdict(float)
        for k, v in Q.items():
            self.Q[k] = v

    def write_qmatrix(self, filename: str) -> None:
        with open(f"models/{filename}.van", "w") as f:
            f.write(str({k: v for k, v in self.Q.items() if v}))

    def get_optimal_action(self, board: game.Board) -> int:
        # returns an index
        h = board.hash()
        mvs = list(range(self.dim))
        random.shuffle(mvs)
        move = max(mvs, key=lambda x: self.Q[(h, x)])
        while board.grid[move]:  # invalid move
            self.Q[(h, move)] -= 100
            move = max(mvs, key=lambda x: self.Q[(h, x)])
        return move

    def bellman(
        self,
        square_completed: bool,
        other_Q: "QTable",
        board: game.Board,
        action: int,
        new_board: game.Board,
        reward: float,
        lr: float,
        discount: float,
        game_over: bool,
    ) -> None:

        board_hash = board.hash()
        if game_over:
            self.Q[(board_hash, action)] = reward
            return
        new_board_hash = new_board.hash()

        if square_completed:  # square completed, so it's still the same QTable's turn
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

            new_new_board = new_board.copy()
            completed, winner = True, 0
            while completed and not winner:
                other_best_action = other_Q.get_optimal_action(new_new_board)
                completed = new_new_board.player_add_edge(
                    new_new_board.index_to_move(other_best_action),
                    other_Q.player,
                )
                winner = new_new_board.is_game_over()

            if not winner:
                new_new_board_hash = new_new_board.hash()
                self.Q[(board_hash, action)] = (1 - lr) * self.Q[
                    (board_hash, action)
                ] + lr * (
                    reward
                    + discount
                    * max(self.Q[(new_new_board_hash, a)] for a in range(self.dim))
                )
            else:
                # game is over by other person's moves
                if winner == 3:
                    rew = TIE_REWARD
                elif winner == self.player:
                    rew = WINNER_REWARD
                else:
                    rew = LOSER_REWARD
                self.Q[(board_hash, action)] = (1 - lr) * self.Q[
                    (board_hash, action)
                ] + lr * (reward + discount * rew)

    def game_player(self, board: game.Board, exploration: float = 0.0) -> game.Move:
        if random.random() <= exploration:
            move = random.randint(0, self.dim - 1)
            while not board.is_valid(board.index_to_move(move)):
                self.Q[(board.hash(), move)] -= 100
                move = random.randint(0, self.dim - 1)
        else:
            move = self.get_optimal_action(board)
        return board.index_to_move(move)


def train_qtables(
    num_rows: int,
    num_cols: int,
    epochs: int = 1000,
    initial_exploration: float = 1,
    initial_lr=0.1,
    discount=1,
    verbose=True,
    model1: QTable = None,
    model2: QTable = None,
    train_model1: bool = True,
    train_model2: bool = True,
) -> Tuple[QTable, QTable]:

    model1 = model1 or QTable(num_rows, num_cols, 1)
    model2 = model2 or QTable(num_rows, num_cols, 2)

    if not train_model1 and not train_model2:
        return model1, model2

    models = {1: model1, 2: model2}
    train_models = {1: train_model1, 2: train_model2}

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
            lambda b: model1.game_player(b, exploration),
            lambda b: model2.game_player(b, exploration),
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
            elif train_models[turn]:  # if we are training this model
                square_completed = cont_game.turn == turn
                models[turn].bellman(
                    square_completed=square_completed,
                    other_Q=models[1 if turn == 2 else 2],
                    board=current_board,
                    action=move_index,
                    new_board=new_board,
                    reward=SQUARE_COMPLETED_REWARD if square_completed else 0,
                    lr=lr,
                    discount=discount,
                    game_over=False,
                )

        # If it is not a tie (winner != 3), then from game.Board.is_game_over, the winner must have
        # # made the last move and completed a square
        if winner != 3 and winner != turn:
            raise ValueError("Something went wrong!!")

        if train_models[turn]:
            models[turn].bellman(
                square_completed=cont_game.turn == turn,
                other_Q=models[1 if turn == 2 else 2],
                board=current_board,
                action=move_index,
                new_board=new_board,
                reward=TIE_REWARD if winner == 3 else WINNER_REWARD,
                lr=lr,
                discount=discount,
                game_over=True,
            )

    return model1, model2


if __name__ == "__main__":
    num_rows, num_cols = 2, 2

    # model1 = QTable(num_rows, num_cols, 1)
    # model1.load_qmatrix(f"{num_rows}{num_cols}_player1")
    # model2 = QTable(num_rows, num_cols, 2)
    # model2.load_qmatrix(f"{num_rows}{num_cols}_player2")

    # game.Game(
    #     num_rows,
    #     num_cols,
    #     player1=model1.game_player,
    #     # player1=game.user_player
    #     # player2=model2.game_player,
    #     player2=game.user_player,
    # ).play()

    model1, model2 = train_qtables(num_rows, num_cols, epochs=1000, verbose=True)
    print(
        "Winner",
        game.Game(
            num_rows,
            num_cols,
            player1=model1.game_player,
            player2=model2.game_player,
        ).play(False),
    )

    for _ in range(4):

        model1, model2 = train_qtables(
            num_rows,
            num_cols,
            epochs=1000,
            model1=model1,
            model2=model2,
            train_model1=False,
            verbose=False,
        )
        print(
            "Winner",
            game.Game(
                num_rows,
                num_cols,
                player1=model1.game_player,
                player2=model2.game_player,
            ).play(False),
            "(should be 1, 2, or 3)",
        )

        model1, model2 = train_qtables(
            num_rows,
            num_cols,
            epochs=1000,
            model1=model1,
            model2=model2,
            train_model2=False,
            verbose=False,
        )
        print(
            "Winner",
            game.Game(
                num_rows,
                num_cols,
                player1=model1.game_player,
                player2=model2.game_player,
            ).play(False),
            "(should be 1)",
        )

    model1, model2 = train_qtables(
        num_rows, num_cols, epochs=1000, model1=model1, model2=model2, verbose=False
    )
    print(
        "Winner",
        game.Game(
            num_rows,
            num_cols,
            player1=model1.game_player,
            player2=model2.game_player,
        ).play(False),
    )

    model1.write_qmatrix(f"{num_rows}{num_cols}_player1")
    model2.write_qmatrix(f"{num_rows}{num_cols}_player2")
