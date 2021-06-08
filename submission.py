import logic
import random
from AbstractPlayers import *
import constants as c
import time
import math

# commands to use for move players. dictionary : Move(enum) -> function(board),
# all the functions {up,down,left,right) receive board as parameter and return tuple of (new_board, done, score).
# new_board is according to the step taken, done is true if the step is legal, score is the sum of all numbers that
# combined in this step.
# (you can see GreedyMovePlayer implementation for example)
commands = {Move.UP: logic.up, Move.DOWN: logic.down,
            Move.LEFT: logic.left, Move.RIGHT: logic.right}


# generate value between {2,4} with probability p for 4
def gen_value(p=PROBABILITY):
    return logic.gen_two_or_four(p)


class GreedyMovePlayer(AbstractMovePlayer):
    """Greedy move player provided to you (no need to change),
    the player receives time limit for a single step and the board as parameter and return the next move that gives
    the best score by looking one step ahead.
    """
    def get_move(self, board, time_limit) -> Move:
        optional_moves_score = {}
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                optional_moves_score[move] = score

        return max(optional_moves_score, key=optional_moves_score.get)


class RandomIndexPlayer(AbstractIndexPlayer):
    """Random index player provided to you (no need to change),
    the player receives time limit for a single step and the board as parameter and return the next indices to
    put 2 randomly.
    """
    def get_indices(self, board, value, time_limit) -> (int, int):
        a = random.randint(0, len(board) - 1)
        b = random.randint(0, len(board) - 1)
        while board[a][b] != 0:
            a = random.randint(0, len(board) - 1)
            b = random.randint(0, len(board) - 1)
        return a, b


# part A
class ImprovedGreedyMovePlayer(AbstractMovePlayer):
    """Improved greedy Move Player,
    implement get_move function with greedy move that looks only one step ahead with heuristic.
    (you can add helper functions as you want).
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        optional_moves_score = {}
        open_spaces_move_score = {}

        for move in Move:
            new_board, done, score = commands[move](board)
            open_spaces_move_score[move] = 0
            for i in range(c.GRID_LEN):
                for j in range(c.GRID_LEN):
                    if new_board[i][j] == 0:
                        open_spaces_move_score[move] += 1
            if done:
                heuristic_value = heuristic(new_board)
                optional_moves_score[move] = score
                optional_moves_score[move] = optional_moves_score[move] + heuristic_value + open_spaces_move_score[move]

        return max(optional_moves_score, key=optional_moves_score.get)

    # TODO: add here helper functions in class, if needed


# part B
class MiniMaxMovePlayer(AbstractMovePlayer):
    """MiniMax Move Player,
    implement get_move function according to MiniMax algorithm
    (you can add helper functions as you want).
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        curr_depth = 1
        start_time = time.time()
        optional_moves_score = {}
        best_move = Move.UP
        safety = time_limit * 0.1
        time_diff = time.time() - start_time
        while time_diff < time_limit - safety:
            for move in Move:
                new_board, done, score = commands[move](board)
                if done:
                    optional_moves_score[move] = self.score_index_calculate(new_board, curr_depth, start_time, time_limit)
            best_move = max(optional_moves_score, key=optional_moves_score.get)
            curr_depth += 1
            optional_moves_score.clear()
            time_diff = time.time() - start_time

        return best_move

    # TODO: add here helper functions in class, if needed

    def score_calculate(self, board, depth, start_time, time_limit):
        if depth == 0:
            return calculate_score(board) + heuristic(board)

        safety = time_limit * 0.1
        best_score = -1
        for move in Move:
            time_diff = time.time() - start_time
            if time_diff > time_limit - safety:
                break
            new_board, done, score = commands[move](board)
            if done is True:
                move_score = self.score_index_calculate(new_board, depth - 1, start_time, time_limit)
                if best_score < move_score:
                    best_score = move_score
        return best_score

    def score_index_calculate(self, board, depth, start_time, time_limit) -> int:
        if depth == 0:
            return calculate_score(board) + heuristic(board)

        move_score = []
        safety = time_limit * 0.1
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                time_diff = time.time() - start_time
                if time_diff > time_limit - safety:
                    break
                if board[i][j] == 0:
                    new_board = copyBoard(board)
                    new_board[i][j] = 2
                    move_score.append(self.score_calculate(new_board, depth - 1, start_time, time_limit))
        return min(move_score, default=-1)


class MiniMaxIndexPlayer(AbstractIndexPlayer):
    """MiniMax Index Player,
    this player is the opponent of the move player and need to return the indices on the board where to put 2.
    the goal of the player is to reduce move player score.
    implement get_indices function according to MiniMax algorithm, the value in minimax player value is only 2.
    (you can add helper functions as you want).
    """
    def __init__(self):
        AbstractIndexPlayer.__init__(self)
        # TODO: add here if needed

    def get_indices(self, board, value, time_limit) -> (int, int):
        # TODO: erase the following line and implement this function.
        # initialization:
        optimal_i = 0
        optimal_j = 0
        curr_depth = 1
        start_time = time.time()
        optional_indexes_score = []
        safety = time_limit * 0.1

        time_diff = time.time() - start_time
        while time_diff < time_limit - safety:
            optional_indexes_score.clear()
            for i in range(c.GRID_LEN):
                for j in range(c.GRID_LEN):
                    if time_diff > time_limit - safety:
                        break
                    if board[i][j] == 0:
                        new_board = copyBoard(board)
                        new_board[i][j] = 2
                        curr_index_score = self.score_move_calculate(new_board, curr_depth, start_time, time_limit)
                        optional_indexes_score.append(IndexMove(curr_index_score, i, j))
            if time_diff > time_limit - safety:  # time up! calculation was not accurate in the last depth
                break
            minimal_score = optional_indexes_score[0].score
            for place in range(len(optional_indexes_score)):
                if optional_indexes_score[place].score <= minimal_score:
                    minimal_score = optional_indexes_score[place].score
                    optimal_i = optional_indexes_score[place].i
                    optimal_j = optional_indexes_score[place].j
            curr_depth += 1
            time_diff = time.time() - start_time

        return optimal_i, optimal_j

    def score_calculate(self, board, depth, start_time, time_limit) -> int:
        if depth == 0:
            return calculate_score(board) + heuristic(board)

        move_score = []
        safety = time_limit * 0.1
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                time_diff = time.time() - start_time
                if time_diff > time_limit - safety:
                    break
                if board[i][j] == 0:
                    new_board = copyBoard(board)
                    new_board[i][j] = 2
                    move_score.append(self.score_move_calculate(new_board, depth - 1, start_time, time_limit))
        return min(move_score, default=-1)

    def score_move_calculate(self, board, depth, start_time, time_limit):
        if depth == 0:
            return calculate_score(board) + heuristic(board)

        safety = time_limit * 0.1
        best_score = -1
        for move in Move:
            time_diff = time.time() - start_time
            if time_diff > time_limit - safety:
                break
            new_board, done, score = commands[move](board)
            if done is True:
                move_score = self.score_calculate(new_board, depth - 1, start_time, time_limit)
                if best_score < move_score:
                    best_score = move_score
        return best_score


class IndexMove(object):

    def __init__(self, score, i, j):
        object.__init__(self)
        self.score = score
        self.i = i
        self.j = j
        # TODO: add here if needed


# part C
class ABMovePlayer(AbstractMovePlayer):
    """Alpha Beta Move Player,
    implement get_move function according to Alpha Beta MiniMax algorithm
    (you can add helper functions as you want)
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        self.depth_sum = 0
        self.moves_counter = 0
        # TODO: add here if needed

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        curr_depth = 1
        start_time = time.time()
        optional_moves_score = {}
        best_move = Move.UP
        safety = time_limit * 0.1
        time_diff = time.time() - start_time
        while time_diff < time_limit - safety:
            alpha = -math.inf
            beta = math.inf
            best_score = 0
            for move in Move:
                new_board, done, score = commands[move](board)
                if done:
                    move_score = self.score_index_calculate(new_board, curr_depth, start_time, time_limit, alpha, beta)
                    optional_moves_score[move] = move_score
                    best_score = max(move_score, best_score)
                    alpha = max(alpha, best_score)
            best_move = max(optional_moves_score, key=optional_moves_score.get)
            curr_depth += 1
            optional_moves_score.clear()
            time_diff = time.time() - start_time

        self.depth_sum += curr_depth - 1
        self.moves_counter += 1
        return best_move

    def get_average_depth(self):
        return self.depth_sum / self.moves_counter

        # TODO: add here helper functions in class, if needed

    def score_calculate(self, board, depth, start_time, time_limit, alpha, beta):
        if depth == 0:
            return calculate_score(board) + heuristic(board)

        safety = time_limit * 0.1
        best_score = -1
        for move in Move:
            time_diff = time.time() - start_time
            if time_diff > time_limit - safety:
                break
            new_board, done, score = commands[move](board)
            if done is True:
                move_score = self.score_index_calculate(new_board, depth - 1, start_time, time_limit, alpha, beta)
                if best_score < move_score:
                    best_score = move_score
                alpha = max(alpha, best_score)
                if best_score >= beta:
                    return math.inf

        return best_score

    def score_index_calculate(self, board, depth, start_time, time_limit, alpha, beta):
        if depth == 0:
            return calculate_score(board) + heuristic(board)

        min_score = math.inf
        move_score = []
        safety = time_limit * 0.1
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                time_diff = time.time() - start_time
                if time_diff > time_limit - safety:
                    break
                if board[i][j] == 0:
                    new_board = copyBoard(board)
                    new_board[i][j] = 2
                    curr_score = self.score_calculate(new_board, depth - 1, start_time, time_limit, alpha, beta)
                    move_score.append(curr_score)
                    min_score = min(min_score, curr_score)
                    beta = min(min_score, beta)
                    if min_score <= alpha and min_score != 0:
                        return -math.inf

        return min(move_score, default=-1)


# part D
class ExpectimaxMovePlayer(AbstractMovePlayer):
    """Expectimax Move Player,
    implement get_move function according to Expectimax algorithm.
    (you can add helper functions as you want)
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        curr_depth = 1
        start_time = time.time()
        optional_moves_score = {}
        best_move = Move.UP
        safety = time_limit * 0.1
        time_diff = time.time() - start_time
        while time_diff < time_limit - safety:
            for move in Move:
                new_board_1, done, score = commands[move](board)
                new_board_2 = copyBoard(new_board_1)
                if done:
                    move_score = 0.9 * self.score_index_calculate(new_board_1, curr_depth, start_time, time_limit, 2) +\
                                 0.1 * self.score_index_calculate(new_board_2, curr_depth, start_time, time_limit, 4)
                    optional_moves_score[move] = move_score
            best_move = max(optional_moves_score, key=optional_moves_score.get)
            curr_depth += 1
            optional_moves_score.clear()
            time_diff = time.time() - start_time

        return best_move

    # TODO: add here helper functions in class, if needed

    def score_calculate(self, board, depth, start_time, time_limit):
        if depth == 0:
            return calculate_score(board) + heuristic(board)

        safety = time_limit * 0.1
        best_score = -1
        for move in Move:
            time_diff = time.time() - start_time
            if time_diff > time_limit - safety:
                break
            new_board_1, done, score = commands[move](board)
            new_board_2 = copyBoard(new_board_1)
            if done is True:
                move_score = 0.9 * self.score_index_calculate(new_board_1, depth - 1, start_time, time_limit, 2) + \
                             0.1 * self.score_index_calculate(new_board_2, depth - 1, start_time, time_limit, 4)
                if best_score < move_score:
                    best_score = move_score
        return best_score

    def score_index_calculate(self, board, depth, start_time, time_limit, value) -> int:
        if depth == 0:
            return calculate_score(board) + heuristic(board)

        move_score = []
        safety = time_limit * 0.1
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                time_diff = time.time() - start_time
                if time_diff > time_limit - safety:
                    break
                if board[i][j] == 0:
                    new_board = copyBoard(board)
                    new_board[i][j] = value
                    move_score.append(self.score_calculate(new_board, depth - 1, start_time, time_limit))
        return min(move_score, default=-1)


class ExpectimaxIndexPlayer(AbstractIndexPlayer):
    """Expectimax Index Player
    implement get_indices function according to Expectimax algorithm, the value is number between {2,4}.
    (you can add helper functions as you want)
    """
    def __init__(self):
        AbstractIndexPlayer.__init__(self)
        # TODO: add here if needed

    def get_indices(self, board, value, time_limit) -> (int, int):
        # TODO: erase the following line and implement this function.
        # initialization:
        optimal_i = 0
        optimal_j = 0
        curr_depth = 1
        start_time = time.time()
        optional_indexes_score = []
        safety = time_limit * 0.1

        time_diff = time.time() - start_time
        while time_diff < time_limit - safety:
            optional_indexes_score.clear()
            for i in range(c.GRID_LEN):
                for j in range(c.GRID_LEN):
                    if time_diff > time_limit - safety:
                        break
                    if board[i][j] == 0:
                        new_board = copyBoard(board)
                        new_board[i][j] = value
                        curr_index_score = self.score_move_calculate(new_board, curr_depth, start_time, time_limit)
                        optional_indexes_score.append(IndexMove(curr_index_score, i, j))
            if time_diff > time_limit - safety:  # time up! calculation was not accurate in the last depth
                break
            minimal_score = optional_indexes_score[0].score
            for place in range(len(optional_indexes_score)):
                if optional_indexes_score[place].score <= minimal_score:
                    minimal_score = optional_indexes_score[place].score
                    optimal_i = optional_indexes_score[place].i
                    optimal_j = optional_indexes_score[place].j
            curr_depth += 1
            time_diff = time.time() - start_time

        return optimal_i, optimal_j

    def score_calculate(self, board, depth, start_time, time_limit, value) -> int:
        if depth == 0:
            return calculate_score(board) + heuristic(board)

        move_score = []
        safety = time_limit * 0.1
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                time_diff = time.time() - start_time
                if time_diff > time_limit - safety:
                    break
                if board[i][j] == 0:
                    new_board = copyBoard(board)
                    new_board[i][j] = value
                    move_score.append(self.score_move_calculate(new_board, depth - 1, start_time, time_limit))
        return min(move_score, default=-1)

    def score_move_calculate(self, board, depth, start_time, time_limit):
        if depth == 0:
            return calculate_score(board) + heuristic(board)

        safety = time_limit * 0.1
        best_score = -1
        for move in Move:
            time_diff = time.time() - start_time
            if time_diff > time_limit - safety:
                break
            new_board_1, done, score = commands[move](board)
            new_board_2 = copyBoard(new_board_1)
            if done is True:
                move_score = 0.9 * self.score_calculate(new_board_1, depth - 1, start_time, time_limit, 2) + \
                             0.1 * self.score_calculate(new_board_2, depth - 1, start_time, time_limit, 4)
                if best_score < move_score:
                    best_score = move_score
        return best_score


# Tournament
class ContestMovePlayer(AbstractMovePlayer):
    """Contest Move Player,
    implement get_move function as you want to compete in the Tournament
    (you can add helper functions as you want)
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        curr_depth = 1
        start_time = time.time()
        optional_moves_score = {}
        best_move = Move.UP
        safety = time_limit * 0.1
        time_diff = time.time() - start_time
        while time_diff < time_limit - safety:
            for move in Move:
                new_board, done, score = commands[move](board)
                if done:
                    optional_moves_score[move] = self.score_index_calculate(new_board, curr_depth, start_time, time_limit)
            best_move = max(optional_moves_score, key=optional_moves_score.get)
            curr_depth += 1
            optional_moves_score.clear()
            time_diff = time.time() - start_time

        return best_move

    # TODO: add here helper functions in class, if needed

    def score_calculate(self, board, depth, start_time, time_limit):
        if depth == 0:
            return calculate_score(board) + heuristic(board)

        safety = time_limit * 0.1
        best_score = -1
        for move in Move:
            time_diff = time.time() - start_time
            if time_diff > time_limit - safety:
                break
            new_board, done, score = commands[move](board)
            if done is True:
                move_score = self.score_index_calculate(new_board, depth - 1, start_time, time_limit)
                if best_score < move_score:
                    best_score = move_score
        return best_score

    def score_index_calculate(self, board, depth, start_time, time_limit) -> int:
        if depth == 0:
            return calculate_score(board) + heuristic(board)

        move_score = []
        safety = time_limit * 0.1
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                time_diff = time.time() - start_time
                if time_diff > time_limit - safety:
                    break
                if board[i][j] == 0:
                    new_board = copyBoard(board)
                    new_board[i][j] = 2
                    move_score.append(self.score_calculate(new_board, depth - 1, start_time, time_limit))
        return min(move_score, default=-1)


def calculate_score(board):
    score = 0
    for i in range(c.GRID_LEN):
        for j in range(c.GRID_LEN):
            if board[i][j] != 0:
                score += board[i][j] * (math.log(board[i][j], 2) - 1)
    return score


def heuristic(board):
    colu_score = col_heuristic(board)
    row_score = col_heuristic(logic.transpose(board))
    return max(colu_score, row_score)


def col_heuristic(mat):
    score = 0

    for i in range(c.GRID_LEN):

        for j in range(c.GRID_LEN-1):
            if mat[i][j] == mat[i][j+1] and mat[i][j] != 0:
                mat[i][j] *= 2
                mat[i][j+1] = 0
                score += mat[i][j]

    return score


def copyBoard(board):
    new_board = []
    for i in range(c.GRID_LEN):
        new_board.append([0] * c.GRID_LEN)
    for i in range(c.GRID_LEN):
        for j in range(c.GRID_LEN):
            new_board[i][j] = board[i][j]
    return new_board
