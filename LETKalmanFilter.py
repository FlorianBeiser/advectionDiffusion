"""
Kalman filter update for advection diffusion example.
"""

from ETKalmanFilter import ETKalman
import numpy as np

class LETKalman:
    def __init__(self, statistics, observation, scale_r):
        self.statistics = statistics

        # Model error cov matrix
        self.epsilon = self.statistics.simulator.noise
        
        # Observation and obs error cov matrices
        self.H = observation.H
        self.tau = observation.noise

        # More detailed information on the observation sites
        self.N_y = observation.N_y
        self.observation_positions = observation.positions *\
            np.array([self.statistics.simulator.grid.dx,self.statistics.simulator.grid.dy])

        # Local kernels around observations sites
        self.initializeLocalisation(scale_r)


    def initializeLocalisation(self, scale_r):

        dx = self.statistics.simulator.grid.dx
        dy = self.statistics.simulator.grid.dy
        nx = self.statistics.simulator.grid.nx
        ny = self.statistics.simulator.grid.ny

        self.all_Ls = [None]*self.N_y
        self.all_xrolls = np.zeros(self.N_y, dtype=np.int)
        self.all_yrolls = np.zeros(self.N_y, dtype=np.int)

        for d in range(self.N_y):
            # Collecting rolling information (xroll and yroll are 0)
            self.all_Ls[d], self.all_xrolls[d], self.all_yrolls[d] = \
                LETKalman.getLocalIndices(self.observation_positions[d], scale_r, \
                        dx, dy, nx, ny)

        self.W_loc = LETKalman.getLocalWeightShape(scale_r, dx, dy, nx, ny)

        self.W_forecast = LETKalman.getCombinedWeights(self.observation_positions, scale_r, dx, dy, nx, ny, self.W_loc)


    @staticmethod
    def getLocalIndices(obs_loc, scale_r, dx, dy, nx, ny):
        """ 
        Defines mapping from global domain (nx times ny) to local domain
        """

        boxed_r = dx*scale_r*1.5
        
        localIndices = np.array([[False]*nx]*ny)
        
        #print(obs_loc_cellID)
        loc_cell_left  = int((obs_loc[0]-boxed_r   )//dx)
        loc_cell_right = int((obs_loc[0]+boxed_r+dx)//dx)
        loc_cell_down  = int((obs_loc[1]-boxed_r   )//dy)
        loc_cell_up    = int((obs_loc[1]+boxed_r+dy)//dy)

        xranges = []
        yranges = []
        
        xroll = 0
        yroll = 0

        if loc_cell_left < 0:
            xranges.append((nx+loc_cell_left , nx))
            xroll = loc_cell_left   # negative number
            loc_cell_left = 0 
        elif loc_cell_right > nx:
            xranges.append((0, loc_cell_right - nx))
            xroll = loc_cell_right - nx   # positive number
            loc_cell_right = nx 
        xranges.append((loc_cell_left, loc_cell_right))

        if loc_cell_down < 0:
            yranges.append((ny+loc_cell_down , ny))
            yroll = loc_cell_down   # negative number
            loc_cell_down = 0 
        elif loc_cell_up > ny:
            yranges.append((0, loc_cell_up - ny ))
            yroll = loc_cell_up - ny   # positive number
            loc_cell_up = ny
        yranges.append((loc_cell_down, loc_cell_up))

        for xrange in xranges:
            for yrange in yranges:
                localIndices[yrange[0] : yrange[1], xrange[0] : xrange[1]] = True

                for y in range(yrange[0],yrange[1]):
                    for x in range(xrange[0], xrange[1]):
                        loc = np.array([(x+0.5)*dx, (y+0.5)*dy])

        return localIndices, xroll, yroll


    @staticmethod
    def getLocalWeightShape(scale_r, dx, dy, nx, ny):
        """
        Gives a local stencil with weights based on the distGC
        """
        
        local_nx = int(scale_r*2*1.5)+1
        local_ny = int(scale_r*2*1.5)+1
        weights = np.zeros((local_ny, local_ny))
        
        obs_loc = np.array([local_nx*dx/2, local_ny*dy/2])

        for y in range(local_ny):
            for x in range(local_nx):
                loc = np.array([(x+0.5)*dx, (y+0.5)*dy])
                weights[y,x] = min(1, LETKalman.distGC(obs_loc, loc, scale_r*dx, nx*dx, ny*dy))
                            
        return weights


    @staticmethod
    def distGC(obs, loc, r, lx, ly):
        """
        Calculating the Gasparin-Cohn value for the distance between obs 
        and loc for the localisation radius r.
        
        obs: drifter positions ([x,y])
        loc: current physical location to check (either [x,y] or [[x1,y1],...,[xd,yd]])
        r: localisation scale in the Gasparin Cohn function
        lx: domain extension in x-direction (necessary for periodic boundary conditions)
        ly: domain extension in y-direction (necessary for periodic boundary conditions)
        """
        if not obs.shape == loc.shape: 
            obs = np.tile(obs, (loc.shape[0],1))
        
        if len(loc.shape) == 1:
            dist = min(np.linalg.norm(np.abs(obs-loc)),
                    np.linalg.norm(np.abs(obs-loc) - np.array([lx,0 ])),
                    np.linalg.norm(np.abs(obs-loc) - np.array([0 ,ly])),
                    np.linalg.norm(np.abs(obs-loc) - np.array([lx,ly])) )
        else:
            dist = np.linalg.norm(obs-loc, axis=1)

        # scalar case
        if isinstance(dist, float):
            distGC = 0.0
            if dist/r < 1: 
                distGC = 1 - 5/3*(dist/r)**2 + 5/8*(dist/r)**3 + 1/2*(dist/r)**4 - 1/4*(dist/r)**5
            elif dist/r >= 1 and dist/r < 2:
                distGC = 4 - 5*(dist/r) + 5/3*(dist/r)**2 + 5/8*(dist/r)**3 -1/2*(dist/r)**4 + 1/12*(dist/r)**5 - 2/(3*(dist/r))
        # vector case
        else:
            distGC = np.zeros_like(dist)
            for i in range(len(dist)):
                if dist[i]/r < 1: 
                    distGC[i] = 1 - 5/3*(dist[i]/r)**2 + 5/8*(dist[i]/r)**3 + 1/2*(dist[i]/r)**4 - 1/4*(dist[i]/r)**5
                elif dist[i]/r >= 1 and dist[i]/r < 2:
                    distGC[i] = 4 - 5*(dist[i]/r) + 5/3*(dist[i]/r)**2 + 5/8*(dist[i]/r)**3 -1/2*(dist[i]/r)**4 + 1/12*(dist[i]/r)**5 - 2/(3*(dist[i]/r))

        return distGC
    
    
    @staticmethod
    def getCombinedWeights(observation_positions, scale_r, dx, dy, nx, ny, W_loc):
    
        W_scale = np.zeros((ny, nx))
        
        num_drifters = observation_positions.shape[0]
        #print('found num_drifters:', num_drifters)
        if observation_positions.shape[1] != 2:
            print('observation_positions has wrong shape')
            return None

        # Get the shape of the local weights (drifter independent)
        W_loc = LETKalman.getLocalWeightShape(scale_r, dx, dy, nx, ny)
        
        for d in range(num_drifters):
            # Get local mapping for drifter 
            L, xroll, yroll = LETKalman.getLocalIndices(observation_positions[d,:], scale_r, dx, dy, nx, ny)

            # Roll weigths according to periodic boundaries
            W_loc_d = np.roll(np.roll(W_loc, shift=yroll, axis=0 ), shift=xroll, axis=1)
            
            # Add weights to global domain based on local mapping:
            W_scale[L] += W_loc_d.flatten()

        return W_scale



    def filter(self, ensemble, obs, series=None):

        N_e = ensemble.shape[1]

        X_f_mean = np.average(ensemble, axis=1)
        X_f_pert = ensemble - np.reshape(X_f_mean, (self.statistics.simulator.grid.N_x,1))

        X_a = np.zeros_like(ensemble)

        # Prepare local ETKF analysis
        N_x_local = self.W_loc.shape[0]*self.W_loc.shape[1] 
        X_f_loc_tmp = np.zeros((N_e, N_x_local))
        X_f_loc_pert_tmp = np.zeros((N_e, 1, N_x_local))
        X_f_loc_mean_tmp = np.zeros((1, N_x_local))
            
        X_f_loc = np.zeros((N_x_local, N_e))
        X_f_loc_pert = np.zeros((N_x_local, N_e))


        # Loop over all d
        for d in range(self.N_y):
    
            L, xroll, yroll = self.all_Ls[d], self.all_xrolls[d], self.all_yrolls[d]










        Rinv = np.linalg.inv(self.tau)

        HX_f =  self.H @ ensemble
        HX_f_mean = np.average(HX_f, axis=1)
        HX_f_pert = HX_f - np.reshape(HX_f_mean, (len(obs),1))

        D = obs - HX_f_mean

        A1 = (self.statistics.ensemble.N_e-1)*np.eye(self.statistics.ensemble.N_e)
        A2 = np.dot(HX_f_pert.T, np.dot(Rinv, HX_f_pert))
        A = A1 + A2
        P = np.linalg.inv(A)

        K = np.dot(X_f_pert, np.dot(P, np.dot(HX_f_pert.T, Rinv)))

        X_a_mean = X_f_mean + np.dot(K, D)
        sigma, V = np.linalg.eigh( (self.statistics.ensemble.N_e - 1) * P )

        X_a_pert = np.dot( X_f_pert, np.dot( V, np.dot( np.diag( np.sqrt( np.real(sigma) ) ), V.T )))

        X_a = X_a_pert + np.reshape(X_a_mean, (self.statistics.simulator.grid.N_x,1))

        self.statistics.set_ensemble(X_a)

        return X_a