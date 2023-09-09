"""
Code within all the 'COMPLETE THIS METHOD/TODO' sections was written by Xingchen (Estella) Ye.
"""

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


class Gridworld_HMM:
    def __init__(self, size, epsilon: float = 0, walls: bool = False):
        if walls:
            self.grid = np.zeros(size)
            for cell in walls:
                self.grid[cell] = 1
        else:
            self.grid = np.random.randint(2, size=size)

        self.epsilon = epsilon
        self.trans = self.initT()
        self.obs = self.initO()

    def neighbors(self, cell):
        i, j = cell
        M, N = self.grid.shape
        adjacent = [
            (i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j - 1), 
            (i, j),(i, j + 1), (i + 1, j - 1), (i + 1, j), (i + 1, j + 1),
        ]
        neighbors = []
        for a in adjacent:
            if a[0] >= 0 and a[0] < M and a[1] >= 0 and a[1] < N and self.grid[a] == 0:
                neighbors.append(a)
        return neighbors

    """
    4.1 Transition and observation probabilities
    """

    def initT(self):
        """
        Create and return NxN transition matrix, where N = size of grid.
        """
        M, N = self.grid.shape
        T = np.zeros((M * N, M * N))
        # TODO:
        S = set()
        for i in range(M): #row
          for j in range(N): #col
            if self.grid[i, j] != 1:
              state = N * i + j
              neighbors = self.neighbors((i, j))
              for n in neighbors:
                next = n[0] * N + n[1]
                T[next, state] = 1.0/ len(neighbors)
                S.add(state)

        for state in S:
          if not np.isclose(T[:, state].sum(), 1):
            print('T not sum up to 1')
        
        return T

    def initO(self):
        """
        Create and return 16xN matrix of observation probabilities, where N = size of grid.
        """
        M, N = self.grid.shape
        O = np.zeros((16, M * N))
        # TODO:
        S = set()
        for i in range(M): #row
          for j in range(N): #col
            if self.grid[i, j] != 1:
              state = N * i + j
          
              obs = ['0', '0', '0', '0']
              # NESW
              if i > 0 and self.grid[i-1, j] == 1:
                obs[0] = '1'
              if j < N-1 and self.grid[i, j+1] == 1:
                obs[1] = '1'            
              if i < M-1 and self.grid[i+1, j] == 1:
                obs[2] = '1'
              if j > 0 and self.grid[i, j-1] == 1:
                obs[3] = '1'

              obs_str = ''.join(obs)
              obs_int = int(obs_str, base=2)
              for x in range(16):
                d = bin(obs_int ^ x).count("1")
                O[x, state] = np.power(1-self.epsilon, 4-d) * np.power(self.epsilon, d)
                    
        for state in S:
          if not np.isclose(O[:, state].sum(), 1):
            print('O not sum up to 1')

        return O

    """
    4.2 Inference: Forward, backward, filtering, smoothing
    """

    def forward(self, alpha: npt.ArrayLike, observation: int):
        """Perform one iteration of the forward algorithm.
        Args:
          alpha (np.ndarray): Current belief state.
          observation (int): Integer representation of bitstring observation.
        Returns:
          np.ndarray: Updated belief state.
        """
        # TODO:
        return self.trans * alpha @ self.obs[observation]

    def backward(self, beta: npt.ArrayLike, observation: int):
        """Perform one iteration of the backward algorithm.
        Args:
          beta (np.ndarray): Current "message" of probabilities.
          observation (int): Integer representation of bitstring observation.
        Returns:
          np.ndarray: Updated message.
        """
        # TODO:
        return self.trans * beta @ self.obs[observation]

    def filtering(self, init: npt.ArrayLike, observations: list[int]):
        """Perform filtering over all observations.
        Args:
          init (np.ndarray): Initial belief state.
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Estimated belief state at each timestep.
        """
        # TODO:
        M, N = self.grid.shape
        filter = np.zeros((N * M, len(observations)))
        arr = init.copy()
        for i in range(len(observations)):
          arr = self.forward(arr, observations[i])
          arr = arr/arr.sum()
          filter[:, i] = arr

        return filter

    def smoothing(self, init: npt.ArrayLike, observations: list[int]):
        """Perform smoothing over all observations.
        Args:
          init (np.ndarray): Initial belief state.
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Smoothed belief state at each timestep.
        """
        # TODO:
        M, N = self.grid.shape
        smooth = np.zeros((N * M, len(observations)))
        alphas = self.filtering(init, observations)
        beta = init.copy()
        for i in range(len(observations)-1, -1, -1):
          beta = self.backward(beta, observations[i])
          arr = alphas[:, i] * beta
          arr = arr/arr.sum()
          smooth[:, i] = arr

        return smooth

    """
    4.3 Localization error
    """

    def loc_error(self, beliefs: npt.ArrayLike, trajectory: list[int]):
        """Compute localization error at each timestep.
        Args:
          beliefs (np.ndarray): Belief state at each timestep.
          trajectory (list[int]): List of states visited.
        Returns:
          list[int]: Localization error at each timestep.
        """
        # TODO:
        M, N = self.grid.shape # M: num_rows, N: num_cols
        loc_errors = []
        for i in range(len(trajectory)):
          actual = trajectory[i]
          predicted = np.argmax(beliefs[:, i])

          actual_x = actual / N
          actual_y = actual % N

          predicted_x = predicted / N
          predicted_y = predicted % N

          error = abs(actual_x - predicted_x) + abs(actual_y - predicted_y)
          loc_errors.append(error)
        return loc_errors




