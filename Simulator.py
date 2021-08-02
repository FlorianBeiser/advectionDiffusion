"""
Forward model in the advection diffusion example.
Ms spatio-temporal model we separate it into the underlying grid 
and the actual model propagation.
"""

import numpy as np
import datetime
from scipy.linalg import circulant 

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

        self.x_grid = np.arange(0,self.nx) * self.dx
        self.y_grid = np.arange(0,self.ny) * self.dy
        self.xx, self.yy = np.meshgrid(self.x_grid, self.y_grid)

        self.xy = np.column_stack( (np.reshape(self.yy, self.N_x),
                                    np.reshape(self.xx, self.N_x)) )[:,[1,0]]

        self.dist_mat = np.eye(self.N_x)
        for i in range(self.N_x):
            self.dist_mat[i,:] = np.linalg.norm(self.xy - self.xy[i], axis=1)



class Simulator:
    def __init__(self, grid, D=0.05, v=[0.5,0.1], zeta=-0.0001, dt=0.01, noise_level=0.1):
        """
        D - diffusion parameter
        v = np.array([v_x,v_y]) - advection 
        zeta - damping parameter
        """
        self.grid = grid
        self.M = self.matrix(D, v, zeta, dt)
        self.noise = self.noise(noise_level)

        self.D = D 
        self.v = v
        self.zeta = zeta
        self.dt = dt

        self.noise_level = noise_level

        self.timestamp = ""


    @staticmethod
    def _neighborsDerivatives(i,ne,N):
        """
        Periodic boundary conditions
        """
        #(under,left,right,over)
        jumps = np.array((-ne, -1, 1, ne))
        
        #under
        if((i - ne) < 0):
            jumps[0]  = ne
        #over
        if((i + ne) > N-1):
            jumps[3] = -ne
        #left
        if((i % ne) == 0):
            jumps[1] = 1
        #right
        if((i % ne) == ne-1):
            jumps[2] = -1
    
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


    def propagate(self, mean, cov=None, steps=1):
        for t in range(steps):
            mean = np.matmul(self.M, mean)
            if cov is not None:
                cov = np.matmul(self.M,np.matmul(cov, self.M.transpose())) + self.noise

        if cov is not None:
            return (mean, cov)
        else:
            return mean

    
    def noise(self, noise_level):
        return np.diag(noise_level*np.ones(self.grid.N_x))


    def to_file(self):
        self.timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        file = "truth_" + self.timestamp

        f = open(file, "x")
        f.write("--------------------------------------------\n")
        f.write("Truth for the advection diffusion experiment\n")
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
        f.write("simulator.noise_level = " + str(self.noise_level) + "\n")
        f.close()


def from_file(timestamp):
    f = open("truth_"+timestamp, "r")
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
    noise_level = float(f.readline()[24:-1])

    simulator = Simulator(grid, D, v, zeta, dt, noise_level)
    simulator.timestamp = timestamp

    f.close()

    return grid, simulator
