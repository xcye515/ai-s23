"""
Code within all the 'COMPLETE THIS METHOD/TODO' sections was written by Xingchen (Estella) Ye.
"""

import random

import numpy as np
import numpy.typing as npt

from hw2.utils import utility, successors, Node, Tree, GameStrategy


"""
Alpha Beta Search
"""


def max_value(state: npt.ArrayLike, alpha: float, beta: float, k: int):
    """Find the max value given state and return the utility and the game board
    state after the move. Please see Lecture 6 for pseudocode.

    Args:
        state (np.ndarray): the state of the game board, mxn numpy array.
        alpha (float): the alpha value
        beta (float): the beta value
        k (int): the number of consecutive marks
    Returns:
        tuple[float, np.ndarray]: utility value and the board after the move
    """

    # TODO:
    util = utility(state, k)
    if util is not None: 
        return util, None
    
    v = -float('inf')
    for new_state in successors(state, 'X'):
        v2, new_state2 = min_value(new_state, alpha, beta, k)
        if v2 > v:
            v, move = v2, new_state
            alpha = max(alpha, v)
        if v >= beta:
            return v, move

    return v, move


def min_value(state: npt.ArrayLike, alpha: float, beta: float, k: int):
    """Find the min value given state and return the utility and the game board
    state after the move. Please see Lecture 6 for pseudocode.

    Args:
        state (np.ndarray): the state of the game board, mxn numpy array.
        alpha (float): the alpha value
        beta (float): the beta value
        k (int): the number of consecutive marks
    Returns:
        tuple[float, np.ndarray]: utility value and the board after the move
    """

    # TODO:
    util = utility(state, k)
    if util is not None: 
        return util, None
    
    v = +float('inf')
    for new_state in successors(state, 'O'):
        v2, new_state2 = max_value(new_state, alpha, beta, k)
        if v2 < v:
            v, move = v2, new_state
            beta = min(beta, v)
        if v <= alpha:
            return v, move

    return v, move


"""
Monte Carlo Tree Search
"""


def select(tree: "Tree", state: npt.ArrayLike, k: int, alpha: float):
    """Starting from state, find a terminal node or node with unexpanded
    children. If all children of a node are in tree, move to the one with the
    highest UCT value.

    Args:
        tree (utils.Tree): the search tree
        state (np.ndarray): the game board state
        k (int): the number of consecutive marks
        alpha (float): exploration parameter
    Returns:
        np.ndarray: the game board state
    """

    # TODO:
    curr = state
    while utility(curr, k) is None:
        curr_node = tree.get(curr)
        max_uct = -1
        next_state = None
        for new_state in successors(curr, curr_node.player):
            repeat = False
            for node in tree.nodes.values():
                if np.array_equal(new_state, node.state) and not np.array_equal(curr, node.parent.state):
                    repeat = True
                    break
            if repeat is False:     
                child = tree.get(new_state)        
                if child is None:
                    return curr
                if max_uct < child.w/float(child.N):
                    max_uct = child.w/float(child.N)
                    next_state = new_state
        if next_state is None:
            return curr
        else:
            curr = next_state


    return curr


def expand(tree: "Tree", state: npt.ArrayLike, k: int):
    """Add a child node of state into the tree if it's not terminal and return
    tree and new state, or return current tree and state if it is terminal.

    Args:
        tree (utils.Tree): the search tree
        state (np.ndarray): the game board state
        k (int): the number of consecutive marks
    Returns:
        tuple[utils.Tree, np.ndarray]: the tree and the game state
    """

    # TODO:
    if utility(state, k) is not None:
        return tree, state
    node = tree.get(state)

    for new_state in successors(state, node.player):
        child = tree.get(new_state)
        if child is None:
            if node.player == 'X':
                new_p = 'O'
            else:
                new_p = 'X'
            tree.add(Node(state=new_state, parent=node, player=new_p, w=0, N=0))
            return tree, new_state

    return tree, state


def simulate(state: npt.ArrayLike, player: str, k: int):
    """Run one game rollout from state to a terminal state using random
    playout policy and return the numerical utility of the result.

    Args:
        state (np.ndarray): the game board state
        player (string): the player, `O` or `X`
        k (int): the number of consecutive marks
    Returns:
        float: the utility
    """

    # TODO:
    curr = state
    util = utility(curr, k)

    while util is None: 
        curr = random.choice(successors(curr, player))
        if player == 'O':
            player = 'X'
        else:
            player = 'O'
        util = utility(curr, k)
    
    return util


def backprop(tree: "Tree", state: npt.ArrayLike, result: float):
    """Backpropagate result from state up to the root.
    All nodes on path have N, number of plays, incremented by 1.
    If result is a win for a node's parent player, w is incremented by 1.
    If result is a draw, w is incremented by 0.5 for all nodes.

    Args:
        tree (utils.Tree): the search tree
        state (np.ndarray): the game board state
        result (float): the result / utility value

    Returns:
        utils.Tree: the game tree
    """

    # TODO:
    node = tree.get(state)

    while node is not None:
        node.N += 1
        if result == 0:
            node.w += 0.5
        elif result == 1 and node.player == 'O':
            node.w += 1
        elif result == -1 and node.player == 'X':
            node.w += 1

        node = node.parent

    return tree


# ******************************************************************************
# ****************************** ASSIGNMENT ENDS *******************************
# ******************************************************************************


def MCTS(state: npt.ArrayLike, player: str, k: int, rollouts: int, alpha: float):
    # MCTS main loop: Execute MCTS steps rollouts number of times
    # Then return successor with highest number of rollouts
    tree = Tree(Node(state, None, player, 0, 1))

    for i in range(rollouts):
        leaf = select(tree, state, k, alpha)
        tree, new = expand(tree, leaf, k)
        result = simulate(new, tree.get(new).player, k)
        tree = backprop(tree, new, result)

    nxt = None
    plays = 0

    for s in successors(state, tree.get(state).player):
        if tree.get(s).N > plays:
            plays = tree.get(s).N
            nxt = s

    return nxt


def ABS(state: npt.ArrayLike, player: str, k: int):
    # ABS main loop: Execute alpha-beta search
    # X is maximizing player, O is minimizing player
    # Then return best move for the given player
    if player == "X":
        value, move = max_value(state, -float("inf"), float("inf"), k)
    else:
        value, move = min_value(state, -float("inf"), float("inf"), k)

    return value, move


def game_loop(
    state: npt.ArrayLike,
    player: str,
    k: int,
    Xstrat: GameStrategy = GameStrategy.RANDOM,
    Ostrat: GameStrategy = GameStrategy.RANDOM,
    rollouts: int = 0,
    mcts_alpha: float = 0.01,
    print_result: bool = False,
):
    # Plays the game from state to terminal
    # If random_opponent, opponent of player plays randomly, else same strategy as player
    # rollouts and alpha for MCTS; if rollouts is 0, ABS is invoked instead
    current = player
    while utility(state, k) is None:
        if current == "X":
            strategy = Xstrat
        else:
            strategy = Ostrat

        if strategy == GameStrategy.RANDOM:
            state = random.choice(successors(state, current))
        elif strategy == GameStrategy.ABS:
            _, state = ABS(state, current, k)
        else:
            state = MCTS(state, current, k, rollouts, mcts_alpha)

        current = "O" if current == "X" else "X"

        if print_result:
            print(state)

    return utility(state, k)
