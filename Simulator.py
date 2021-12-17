"""
Forward model in the advection diffusion example.
Ms spatio-temporal model we separate it into the underlying grid 
and the actual model propagation.
"""

import numpy as np
import os

from matplotlib import pyplot as plt

from scipy.linalg.special_matrices import toeplitz 

import Sampler

class Grid:
    """Grid for the forward model"""
    def __init__(self, nx, ny, dx, dy):
        self.nx = nx
        self.ny = ny 
        self.N_x = self.nx * self.ny

        self.dx = dx
        self.dy = dy

        self.xdim = self.dx*self.nx
        self.ydim = self.dy*self.ny

        # Auxiliary matrix for the construction of the circullant distance matrix
        self.dist_toepitz = np.zeros((self.ny, self.nx))
        for i in range(self.nx):
            if i <= self.nx/2:
                di = i
            else:
                di = self.nx - i 
            
            for j in range(self.ny):
                if j <= self.ny/2:
                    dj = j 
                else:
                    dj = self.ny - j 

                Dx = di * self.dx
                Dy = dj * self.dy
                self.dist_toepitz[j,i] = np.sqrt(Dx**2 + Dy**2)

        self.dist_mat = np.zeros((self.N_x,self.N_x))
        for j in range(self.ny):
            toeplitz = np.roll(self.dist_toepitz,j,axis=0)
            for i in range(self.nx):
                self.dist_mat[j*self.nx + i] = np.reshape(np.roll(toeplitz,i,axis=1),(1,self.N_x))
    
    def point2idx(self, points):
        if any(isinstance(el, list) for el in points):
            idxs = []
            for point in points:
                indicator_field = np.zeros((self.ny, self.nx))
                indicator_field[point[1],point[0]] = 1.0
                idx = np.where(indicator_field.flatten() != 0 )[0][0]
                idxs.append(idx)
            return idxs

        else:
            indicator_field = np.zeros((self.ny, self.nx))
            indicator_field[points[1],points[0]] = 1.0
            idx = np.where(indicator_field.flatten() != 0 )[0][0]
            return idx




class Simulator:
    def __init__(self, grid, D=0.05, v=[0.5,0.1], zeta=-0.0001, dt=0.01, noise_stddev=0.1, noise_matern_phi=1.0):
        """
        D - diffusion parameter
        v = np.array([v_x,v_y]) - advection 
        zeta - damping parameter
        """
        self.grid = grid
        self.M = self.matrix(D, v, zeta, dt)

        self.D = D 
        self.v = v
        self.zeta = zeta
        self.dt = dt

        self.noise_matern_phi = noise_matern_phi
        self.noise_stddev = noise_stddev

        self.Q = self.cov_matrix()


    @staticmethod
    def _neighborsDerivatives(i,ne,N):
        """
        Periodic boundary conditions
        """
        #(under,left,right,over)
        jumps = np.array((-ne, -1, 1, ne))
        
        #under
        if((i - ne) < 0):
            jumps[0]  = N - ne
        #over
        if((i + ne) > N-1):
            jumps[3] = ne - N 
        #left
        if((i % ne) == 0):
            jumps[1] = ne-1
        #right
        if((i % ne) == ne-1):
            jumps[2] = -(ne-1)
    
        return(jumps+i)


    def matrix(self, D, v, zeta, dt):
        N = self.grid.N_x
        ve = np.repeat(v[0], N)
        vn = np.repeat(v[1], N)
        zeta = np.repeat(zeta, N)
        dx = self.grid.dx
        dy = self.grid.dy
        
        diag_const = zeta -2*D/(dx**2) -2*D/(dy**2)  # main
        diag_minus_1 = -(-ve/(2*dx))+ D/(dx**2)  #left
        diag_plus_1 = (-ve/(2*dx)) + D/(dx**2)   #right 
        diag_minus_N = -(-vn/(2*dy)) + D/(dy**2) #under 
        diag_plus_N = (-vn/(2*dy)) + D/(dy**2)   #over
        
        M = np.diag(diag_const)
    
        for i in range(N):
            neighbors = Simulator._neighborsDerivatives(i,self.grid.nx,N)
            for j in range(4):
                if(j==0):
                    M[i,neighbors[j]] = M[i,neighbors[j]] + diag_minus_N[i]
                if(j==1):
                    M[i,neighbors[j]] = M[i,neighbors[j]] + diag_minus_1[i]
                if(j==2):
                    M[i,neighbors[j]] = M[i,neighbors[j]] + diag_plus_1[i]
                if(j==3):
                    M[i,neighbors[j]] = M[i,neighbors[j]] + diag_plus_N[i]
    
        return (np.diag(np.ones(N)) + dt*M)


    def cov_matrix(self):
        
        noise_args = {"mean_upshift"     : 0.0,
                      "matern_phi"       : self.noise_matern_phi,
                      "stddev"           : self.noise_stddev}

        self.noise_sampler = Sampler.Sampler(self.grid, noise_args)

        return self.noise_sampler.cov


    def propagate(self, mean, cov=None, steps=1):
        for t in range(steps):
            mean = np.matmul(self.M, mean)
            if cov is not None:
                cov = np.matmul(self.M,np.matmul(cov, self.M.transpose())) + self.noise

        if cov is not None:
            return (mean, cov)
        else:
            return mean


    def plot_correlation_points(self, points):
        """Showing point on the grid with advection field"""
        plt.figure()
        plt.xlim(0, self.grid.nx-1)
        plt.ylim(0, self.grid.ny-1)
        plt.suptitle("Points for Correlation Study")
        plt.title("Chosen in Advection Direction")
        freq = self.grid.nx/10
        X, Y = np.meshgrid( np.arange(0,self.grid.nx,freq), np.arange(0,self.grid.ny,freq) )
        plt.quiver(X.flatten(), Y.flatten(), np.repeat(self.v[0],len(X.flatten())), np.repeat(self.v[1],len(Y.flatten())))
        i = 0
        for point in points:
            plt.scatter(point[0], point[1], c="red", s=150)
            plt.text(point[0], point[1], str(i), c="black", fontsize=12)
            i=i+1
        plt.show()



    def to_file(self, timestamp):
        
        root_path = os.getcwd()
        new_path = os.path.join(root_path, "experiment_files")
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        exp_path = os.path.join(new_path, "experiment_" + timestamp)
        os.makedirs(exp_path)

        file = "experiment_files/experiment_" + timestamp + "/setup"

        f = open(file, "a")
        f.write("--------------------------------------------\n")
        f.write("Setup for the advection diffusion experiment\n")
        f.write("--------------------------------------------\n")
        f.write("The grid:\n")
        f.write("grid.nx = " + str(self.grid.nx) + "\n")
        f.write("grid.ny = " + str(self.grid.ny) + "\n")
        f.write("grid.dx = " + str(self.grid.dx) + "\n")
        f.write("grid.dy = " + str(self.grid.dy) + "\n")
        f.write("--------------------------------------------\n")
        f.write("The parameters for the advection diffusion equation:\n")
        f.write("simulator.D = " + str(self.D) + "\n")
        f.write("simulator.v = " + str(self.v) + "\n")
        f.write("simulator.zeta = " + str(self.zeta) + "\n")
        f.write("simulator.dt = " + str(self.dt) + "\n")
        f.write("simulator.noise_stddev = " + str(self.noise_stddev) + "\n")
        f.write("simulator.noise_matern_phi = " + str(self.noise_matern_phi) + "\n")
        f.close()


def from_file(timestamp):
    f = open("experiment_files/experiment_"+timestamp+"/setup", "r")
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    nx = int(f.readline()[10:-1])
    ny = int(f.readline()[10:-1])
    dx = float(f.readline()[10:-1])
    dy = float(f.readline()[10:-1])
    
    grid = Grid(nx,ny,dx,dy)

    f.readline()
    f.readline()
    D = float(f.readline()[14:-1])
    v = f.readline()[14:-1].strip('][').split(', ')
    v[0] = float(v[0])
    v[1] = float(v[1])
    zeta = float(f.readline()[17:-1])
    dt = float(f.readline()[15:-1])
    noise_stddev = float(f.readline()[25:-1])
    noise_matern_phi = float(f.readline()[29:-1])

    simulator = Simulator(grid, D=D, v=v, zeta=zeta, dt=dt, 
                            noise_stddev=noise_stddev, noise_matern_phi=noise_matern_phi)

    f.close()

    return grid, simulator
