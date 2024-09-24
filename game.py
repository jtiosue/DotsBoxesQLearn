from typing import Tuple, Callable
import numpy as np
import random


class Move:
    """
    Holds row, col, and vert.
    vert = True denotes a vertical line, False a horizontal line.
    Then row and col denote the row and column of the line (zero indexed)
    """

    def __init__(self, row: int, col: int, vert: bool) -> None:
        self.row, self.col, self.vert = row, col, vert
        self.valid = True


class Board:
    def __init__(self, num_rows: int, num_cols: int) -> None:
        self.num_rows, self.num_cols = num_rows, num_cols
        self.num_vert_edges = num_rows * (num_cols + 1)
        self.num_hori_edges = num_cols * (num_rows + 1)
        self.grid = np.zeros(self.num_vert_edges + self.num_hori_edges)
        self.squares = np.zeros((num_rows, num_cols), dtype=np.int8)
        # self.squares = {(r, c): None for r in range(num_rows) for c in range(num_cols)}

    def copy(self):
        b = Board(self.num_rows, self.num_cols)
        b.grid = self.grid.copy()
        b.squares = self.squares.copy()
        return b

    def reset(self) -> None:
        self.grid = np.zeros(self.num_vert_edges + self.num_hori_edges)
        self.squares = np.zeros((self.num_rows, self.num_cols))
        # self.squares = {
        #     (r, c): None for r in range(self.num_rows) for c in range(self.num_cols)
        # }

    def move_to_index(self, move: Move) -> int:
        # unique integer corresponding to an edge
        if move.vert:
            return move.row * (self.num_cols + 1) + move.col
        return self.num_vert_edges + move.row * self.num_cols + move.col

    def index_to_move(self, index: int) -> Move:
        vert = index < self.num_vert_edges
        if vert:
            row = index // (self.num_cols + 1)
            col = index - row * (self.num_cols + 1)
        else:
            index -= self.num_vert_edges
            row = index // self.num_cols
            col = index - row * self.num_cols
        return Move(row, col, vert)

    def add_edge(self, move: Move) -> None:
        self.grid[self.move_to_index(move)] = 1

    def get_edge(self, move: Move) -> int:
        if not self.position_valid(move):
            return 0
        return self.grid[self.move_to_index(move)]

    def add_left_square(self, row: int, col: int, player: int) -> None:
        # left of a vert
        if col - 1 >= 0:
            # self.squares[(row, col - 1)] = player
            self.squares[row, col - 1] = player

    def get_left_square(self, row: int, col: int) -> int:
        # return self.squares.get((row, col - 1), 0)
        return self.squares[row, col - 1]

    def add_right_square(self, row: int, col: int, player: int) -> None:
        # right of a vert
        if col < self.num_cols:
            # self.squares[(row, col)] = player
            self.squares[row, col] = player

    def get_right_square(self, row: int, col: int) -> int:
        # return self.squares.get((row, col), 0)
        return self.squares[row, col]

    def add_top_square(self, row: int, col: int, player: int) -> None:
        # top of a horizontal
        if row - 1 >= 0:
            # self.squares[(row - 1, col)] = player
            self.squares[row - 1, col] = player

    def get_top_square(self, row: int, col: int) -> int:
        # return self.squares.get((row - 1, col), 0)
        return self.squares[row - 1, col]

    def add_bot_square(self, row: int, col: int, player: int) -> None:
        # bottom of a horizontal
        if row < self.num_rows:
            # self.squares[(row, col)] = player
            self.squares[row, col] = player

    def get_bot_square(self, row: int, col: int) -> int:
        # return self.squares.get((row, col), 0)
        return self.squares[row, col]

    def square_completed(self, row: int, col: int) -> bool:
        return (
            self.get_edge(Move(row, col, True))
            and self.get_edge(Move(row, col + 1, True))
            and self.get_edge(Move(row, col, False))
            and self.get_edge(Move(row + 1, col, False))
        )

    def position_valid(self, move: Move) -> bool:
        if move.vert:
            return 0 <= move.row < self.num_rows and 0 <= move.col <= self.num_cols
        return 0 <= move.row <= self.num_rows and 0 <= move.col < self.num_cols

    def is_valid(self, move: Move) -> bool:
        return not self.get_edge(move) and self.position_valid(move)

    def player_add_edge(self, move: Move, player: int) -> bool:
        # returns true if player completed a square. False otherwise.
        self.add_edge(move)
        completed = False
        if move.vert:
            if self.square_completed(move.row, move.col - 1):
                self.add_left_square(move.row, move.col, player)
                completed = True
            if self.square_completed(move.row, move.col):
                self.add_right_square(move.row, move.col, player)
                completed = True
        else:
            if self.square_completed(move.row - 1, move.col):
                self.add_top_square(move.row, move.col, player)
                completed = True
            if self.square_completed(move.row, move.col):
                self.add_bot_square(move.row, move.col, player)
                completed = True
        return completed

    def score(self) -> Tuple[int, int]:
        vals = self.squares.flatten()
        return len([x for x in vals if x == 1]), len([x for x in vals if x == 2])

    def is_game_over(self) -> int:
        # return 0 if game is ongoing, 1 if player 1 won, 2 if player 2 won, or 3 if it is a tie
        total_pos_points = self.num_rows * self.num_cols
        p1, p2 = self.score()
        if p1 > total_pos_points / 2:
            return 1
        elif p2 > total_pos_points / 2:
            return 2
        elif np.all(self.grid):
            return 3
        else:
            return 0

    def hash(self) -> Tuple[int, int, int]:
        h = int(np.dot(2 ** np.arange(len(self.grid)), self.grid))
        s = self.score()
        return h, s[0], s[1]

    def __str__(self) -> str:
        s = "Score: " + str(self.score()) + "\n"
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                move = Move(row, col, False)
                s += "*" + ("-" if self.get_edge(move) else " ")
            s += "*\n"
            for col in range(self.num_cols):
                move = Move(row, col, True)
                s += "|" if self.get_edge(move) else " "
                s += str(sq) if (sq := self.get_right_square(row, col)) else " "
            move = Move(row, self.num_cols, True)
            s += "|\n" if self.get_edge(move) else " \n"

        for col in range(self.num_cols):
            move = Move(self.num_rows, col, False)
            s += "*" + ("-" if self.get_edge(move) else " ")
        s += "*\n"

        return s


class Game:
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        player1: Callable[[Board], Move],
        player2: Callable[[Board], Move],
    ) -> None:
        self.board = Board(num_rows, num_cols)
        self.turn = 1
        self.players = {1: player1, 2: player2}

    def reset(self, shuffle: bool = False) -> None:
        self.board.reset()
        self.turn = random.choice([1, 2]) if shuffle else 1

    def play_step(self, verbose: bool = True) -> Move:
        # if the move is invalid, then move.valid will be False
        if verbose:
            print(self.board)
            print(f"Player {self.turn}'s turn")
        move = self.players[self.turn](self.board)
        if not self.board.is_valid(move):
            move.valid = False
            return move
        if not self.board.player_add_edge(move, self.turn):
            self.turn = 2 if self.turn == 1 else 1

        return move

    def play(self, verbose: bool = True) -> int:

        playing, winner = True, 0

        while playing:
            move = self.play_step(verbose)
            if not move.valid:
                winner = 2 if self.turn == 1 else 1
                playing = False
            else:
                playing = not bool(self.board.is_game_over())
                if not playing:
                    winner = self.board.is_game_over()

        if verbose:
            print(self.board)
            print("Winner:", winner)
        return winner


def user_player(board: Board) -> Move:
    try:
        move = Move(*eval(input("Enter move: ")))
    except (SyntaxError, TypeError):
        print("Invalid entry, try again")
        return user_player(board)

    if not board.is_valid(move):
        print("Invalid move, try again")
        return user_player(board)
    return move


if __name__ == "__main__":

    num_rows, num_cols = 2, 2
    Game(num_rows, num_cols, user_player, user_player).play()
