"""
Forward model in the advection diffusion example.
Ms spatio-temporal model we separate it into the underlying grid 
and the actual model propagation.
"""

import numpy as np
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

        self.dist_mat = circulant(np.reshape(self.dist_toepitz, self.N_x))



class Simulator:
    def __init__(self, grid, D=0.05, v=[0.5,0.1], zeta=-0.0001, dt=0.01, noise=0.1):
        """
        D - diffusion parameter
        v = np.array([v_x,v_y]) - advection 
        zeta - damping parameter
        """
        self.grid = grid
        self.M = self.matrix(D, v, zeta, dt)
        self.noise = self.noise(noise)

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


    def forecast(self, mean, cov):
        forecasted_mean = np.matmul(self.M, mean)
        forecasted_covariance = np.matmul(self.M,np.matmul(cov, self.M.transpose())) + self.noise

        return(forecasted_mean, forecasted_covariance)
                
    
    def noise(self, noise):
        return np.diag(noise*np.ones(self.grid.N_x))








