import numpy as np

class Observation:
    """Observations in the advection diffusion example.
    Handling observation values and construction observation operator"""
    def __init__(self, simulator, noise_level=1):
        self.grid = simulator.grid
        self.noise_level = noise_level

        self.N_obs = 0

        print("Remember to set observation positions and to set values!")
        

    def set_positions(self, positions):
        self.positions = positions
        self.N_y = len(positions)
        self.obsidx = np.zeros(self.N_y).astype(int)
        for i in range(self.N_y):
            self.obsidx[i] = positions[i][1] * self.grid.nx + positions[i][0]
        self.matrix()
        self.noise(self.noise_level)

    def matrix(self):
        self.H = np.zeros((self.N_y, self.grid.N_x))
        for i in range(self.N_y):
            self.H[i,self.obsidx[i]] = 1 

    def observe(self, x):
        self.N_obs = self.N_obs + 1 
        obs = np.matmul(self.H,x)
        if self.N_obs == 1:
            self.obses = np.array([obs])
        else:
            self.obses = np.append(self.obses, np.array([obs]), axis=0)
        

    def noise(self, noise_level):
        self.noise = np.diag(np.repeat(noise_level, self.N_y))

    
    def load_observations(self, fname):
        self.obses = np.loadtxt(fname)
        self.N_obs = self.obses.shape[0]
        assert self.obses.shape[1] == self.N_y, "Wrong dimensions!"