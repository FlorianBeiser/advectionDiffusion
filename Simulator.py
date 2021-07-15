"""
Forward model in the advection diffusion example.
Ms spatio-temporal model we separate it into the underlying grid 
and the actual model propagation.
"""

import numpy as np

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
            dist_nn = np.linalg.norm(self.xy - self.xy[i], axis=1)
            dist_nu = np.linalg.norm(self.xy - (self.xy[i] + np.array([0, self.ydim])), axis=1)
            dist_nd = np.linalg.norm(self.xy - (self.xy[i] + np.array([0,-self.ydim])), axis=1)
            dist_ln = np.linalg.norm(self.xy - (self.xy[i] + np.array([-self.xdim,0])), axis=1)
            dist_rn = np.linalg.norm(self.xy - (self.xy[i] + np.array([ self.xdim,0])), axis=1)
            dist_ld = np.linalg.norm(self.xy - (self.xy[i] + np.array([-self.xdim,-self.ydim])), axis=1)
            dist_lu = np.linalg.norm(self.xy - (self.xy[i] + np.array([-self.xdim, self.ydim])), axis=1)
            dist_ru = np.linalg.norm(self.xy - (self.xy[i] + np.array([ self.xdim, self.ydim])), axis=1)
            dist_rd = np.linalg.norm(self.xy - (self.xy[i] + np.array([ self.xdim,-self.ydim])), axis=1)
            dist = np.minimum(dist_nn,dist_nu)
            dist = np.minimum(dist, dist_nd)
            dist = np.minimum(dist, dist_ln)
            dist = np.minimum(dist, dist_rn)
            dist = np.minimum(dist, dist_ld)
            dist = np.minimum(dist, dist_lu)
            dist = np.minimum(dist, dist_ru)
            dist = np.minimum(dist, dist_rd)
            self.dist_mat[i,:] = dist



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








