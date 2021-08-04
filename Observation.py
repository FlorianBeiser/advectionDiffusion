import numpy as np
from matplotlib import pyplot as plt

class Observation:
    """Observations in the advection diffusion example.
    Handling observation values and construction observation operator"""
    def __init__(self, grid, noise_level=0.1):
        self.grid = grid
        self.noise_level = noise_level

        self.N_obs = 0

        print("Remember to set observation positions and to set/observe values!")



    def set_positions(self, positions):
        self.positions = positions
        self.N_y = len(positions)
        self.obsidx = np.zeros(self.N_y).astype(int)
        for i in range(self.N_y):
            self.obsidx[i] = positions[i][1] * self.grid.nx + positions[i][0]
        self.matrix()
        self.noise_matrix()

    
    def plot_positions(self):
        plt.title("Moorings in domain (remember periodic BC)")
        plt.scatter(np.array(self.positions)[:,0],np.array(self.positions)[:,1])
        plt.xlim(0, self.grid.nx)
        plt.ylim(0, self.grid.ny)
        plt.show()


    def matrix(self):
        self.H = np.zeros((self.N_y, self.grid.N_x))
        for i in range(self.N_y):
            self.H[i,self.obsidx[i]] = 1 


    def observe(self, x):
        self.N_obs = self.N_obs + 1 
        obs = self.H @ x + np.random.normal(scale=np.sqrt(self.noise_level), size=self.N_y) # scale = standard deviation
        if self.N_obs == 1:
            self.obses = np.array([obs])
        else:
            self.obses = np.append(self.obses, np.array([obs]), axis=0)


    def noise_matrix(self):
        self.R = self.noise_level * np.eye(self.N_y)

    
    def load_observations(self, fname):
        self.obses = np.loadtxt(fname)
        self.N_obs = self.obses.shape[0]
        assert self.obses.shape[1] == self.N_y, "Wrong dimensions!"

    
    def setup_to_file(self, timestamp):
        file = "experiment_files/experiment_" + timestamp + "/setup"

        f = open(file, "a")
        f.write("--------------------------------------------\n")
        f.write("Properties of the observations:\n")
        f.write("observation.noise_level = " + str(self.noise_level) + "\n")
        f.close()


    def to_file(self, timestamp, obs_timestamp):

        file_positions = "experiment_files/experiment_" + timestamp + "/observation_positions_" + obs_timestamp + ".csv"
        np.savetxt(file_positions, self.positions)
        
        file_values = "experiment_files/experiment_" + timestamp + "/observation_values_" + obs_timestamp + ".csv"
        np.savetxt(file_values, np.reshape(self.obses,(self.N_obs, self.N_y) ))
        
        


def from_file(simulator):
    
    f = open("observation_properties_"+simulator.timestamp, "r")
    f.readline()
    f.readline()
    f.readline()
    noise_level = float(f.readline()[26:-1])
    f.close()

    observation = Observation(simulator, noise_level)

    observation.set_positions(np.loadtxt("observation_positions_"+simulator.timestamp+".csv"))

    observation.load_observations("observation_values_"+simulator.timestamp+".csv")

    return observation
