"""
Kalman filter update for advection diffusion example.
"""

import numpy as np

# DEBUG 
from matplotlib import pyplot as plt

class SLETKalman:
    def __init__(self, statistics, observation, scale_r):
        self.statistics = statistics
        
        # Observation and obs error cov matrices
        self.H = observation.H
        self.R = observation.R

        # More detailed information on the observation sites
        self.N_y = observation.N_y
        self.observation_positions = observation.positions *\
            np.array([self.statistics.simulator.grid.dx,self.statistics.simulator.grid.dy])

        # Grouping for serial processing
        self.initializeGroups(scale_r)

        # Local kernels around observations sites
        self.initializeLocalisation(scale_r)


    def initializeGroups(self, scale_r):
        # Assembling observation distance matrix
        self.obs_dist_mat = np.zeros((self.N_y, self.N_y))
        for i in range(self.N_y):
            for j in range(self.N_y):
                dx = np.abs(self.observation_positions[i][0] - self.observation_positions[j][0])
                if dx > self.statistics.simulator.grid.xdim/2:
                    dx = self.statistics.simulator.grid.xdim - dx
                dy = np.abs(self.observation_positions[i][1] - self.observation_positions[j][1])
                if dy > self.statistics.simulator.grid.ydim/2:
                    dy = self.statistics.simulator.grid.ydim - dy 
                self.obs_dist_mat[i,j] = np.sqrt(dx**2+dy**2)
        # Heavy diagonal such that 0-distances are above every threshold
        np.fill_diagonal(self.obs_dist_mat, np.sqrt(self.statistics.simulator.grid.xdim**2 + self.statistics.simulator.grid.ydim**2))

        # Groups of "un-correlated" observation
        self.groups = list([list(np.arange(self.N_y, dtype=int))])
        # Observations are assumed to be uncorrelated, if distance bigger than threshold
        threshold = 1.5 * scale_r * self.statistics.simulator.grid.dx

        g = 0 
        while self.obs_dist_mat[np.ix_(self.groups[g],self.groups[g])].min() < threshold:
            while self.obs_dist_mat[np.ix_(self.groups[g],self.groups[g])].min() < threshold:
                mask = np.ix_(self.groups[g],self.groups[g])
                idx2move = self.groups[g][np.where(self.obs_dist_mat[mask] == self.obs_dist_mat[mask].min())[1][0]]
                self.groups[g] = list(np.delete(self.groups[g], np.where(self.groups[g] == idx2move)))
                if len(self.groups)<g+2: 
                    self.groups.append([idx2move])
                else:
                    self.groups[g+1].append(idx2move)
            g = g + 1 


    def initializeLocalisation(self, scale_r):

        dx = self.statistics.simulator.grid.dx
        dy = self.statistics.simulator.grid.dy
        nx = self.statistics.simulator.grid.nx
        ny = self.statistics.simulator.grid.ny

        self.W_loc = SLETKalman.getLocalWeightShape(scale_r, dx, dy, nx, ny)

        self.all_Ls = []
        self.all_xrolls = []
        self.all_yrolls = []

        self.W_analyses = []
        self.W_forecasts = []

        for g in range(len(self.groups)):
            Ls = [None]*len(self.groups[g])
            xrolls = np.zeros(len(self.groups[g]), dtype=np.int)
            yrolls = np.zeros(len(self.groups[g]), dtype=np.int)

            for d in range(len(self.groups[g])):
                # Collecting rolling information (xroll and yroll are 0)
                Ls[d], xrolls[d], yrolls[d] = \
                    SLETKalman.getLocalIndices(self.observation_positions[self.groups[g][d]], scale_r, \
                            dx, dy, nx, ny)

            self.all_Ls.append(Ls)
            self.all_xrolls.append(xrolls)
            self.all_yrolls.append(yrolls)

            W_combined = SLETKalman.getCombinedWeights(self.observation_positions[self.groups[g]], scale_r, dx, dy, nx, ny, self.W_loc)

            W_scale    = np.maximum(W_combined,1)

            W_analysis = W_combined/W_scale
            W_forecast = np.ones_like(W_analysis) - W_analysis 

            self.W_analyses.append(W_analysis)
            self.W_forecasts.append(W_forecast)


    @staticmethod
    def getLocalIndices(obs_loc, scale_r, dx, dy, nx, ny):
        """ 
        Defines mapping from global domain (nx times ny) to local domain
        """

        boxed_r = dx*np.ceil(scale_r*1.5)
        
        localIndices = np.array([[False]*nx]*ny)
        
        loc_cell_left  = int(np.round(obs_loc[0]/dx)) - int(np.round(boxed_r/dx))
        loc_cell_right = int(np.round(obs_loc[0]/dx)) + int(np.round((boxed_r+dx)/dx))
        loc_cell_down  = int(np.round(obs_loc[1]/dy)) - int(np.round(boxed_r/dy))
        loc_cell_up    = int(np.round(obs_loc[1]/dy)) + int(np.round((boxed_r+dy)/dy))

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
        
        local_nx = int(np.ceil(scale_r*1.5)*2 + 1)
        local_ny = int(np.ceil(scale_r*1.5)*2 + 1)
        weights = np.zeros((local_ny, local_ny))
        
        obs_loc = np.array([local_nx*dx/2, local_ny*dy/2])

        for y in range(local_ny):
            for x in range(local_nx):
                loc = np.array([(x+0.5)*dx, (y+0.5)*dy])
                weights[y,x] = min(1, SLETKalman.distGC(obs_loc, loc, scale_r*dx, nx*dx, ny*dy))
                            
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
        W_loc = SLETKalman.getLocalWeightShape(scale_r, dx, dy, nx, ny)

        for d in range(num_drifters):
            # Get local mapping for drifter 
            L, xroll, yroll = SLETKalman.getLocalIndices(observation_positions[d,:], scale_r, dx, dy, nx, ny)

            # Roll weigths according to periodic boundaries
            W_loc_d = np.roll(np.roll(W_loc, shift=yroll, axis=0 ), shift=xroll, axis=1)
            
            # Add weights to global domain based on local mapping:
            W_scale[L] += W_loc_d.flatten()

        return W_scale


    def filter_per_group(self, ensemble, obs, g):

        # Bookkeeping
        nx = self.statistics.simulator.grid.nx
        ny = self.statistics.simulator.grid.ny
        N_e = ensemble.shape[1]

        X_f = np.zeros((N_e, ny, nx))
        for e in range(X_f.shape[0]):
            X_f[e] = np.reshape(ensemble[:,e], (ny, nx))

        X_f_mean = np.average(X_f, axis=0)
        X_f_pert = X_f - X_f_mean

        X_a = np.zeros_like(X_f)

        H = self.H[self.groups[g]]
        HX_f =  H @ ensemble
        HX_f_mean = np.average(HX_f, axis=1)
        HX_f_pert = HX_f - np.reshape(HX_f_mean, (len(obs),1))

        # Prepare local ETKF analysis
        N_x_local = self.W_loc.shape[0]*self.W_loc.shape[1] 

        X_f_loc_tmp = np.zeros((N_e, N_x_local))
        X_f_loc_pert_tmp = np.zeros((N_e, N_x_local))
        X_f_loc_mean_tmp = np.zeros((N_x_local))

        # Loop over all d
        for d in range(len(obs)):
    
            L, xroll, yroll = self.all_Ls[g][d], self.all_xrolls[g][d], self.all_yrolls[g][d]

            X_f_loc_tmp[:,:] = X_f[:,L]   
            X_f_loc_pert_tmp[:,:] = X_f_pert[:,L]
            X_f_loc_mean_tmp[:] = X_f_mean[L]

            if not (xroll == 0 and yroll == 0):
                rolling_shape = (N_e, self.W_loc.shape[0], self.W_loc.shape[1]) # roll around axis 2 and 3
                X_f_loc_tmp[:,:] = np.roll(np.roll(X_f_loc_tmp.reshape(rolling_shape), shift=-yroll, axis=1 ), shift=-xroll, axis=2).reshape((N_e, N_x_local))
                X_f_loc_pert_tmp[:,:] = np.roll(np.roll(X_f_loc_pert_tmp.reshape(rolling_shape), shift=-yroll, axis=1 ), shift=-xroll, axis=2).reshape((N_e, N_x_local))

                mean_rolling_shape = (self.W_loc.shape[0], self.W_loc.shape[1]) # roll around axis 1 and 2
                X_f_loc_mean_tmp[:] = np.roll(np.roll(X_f_loc_mean_tmp.reshape(mean_rolling_shape), shift=-yroll, axis=0 ), shift=-xroll, axis=1).reshape((N_x_local))

            # Adapting LETKF dimensionalisation
            X_f_loc = X_f_loc_tmp.T
            X_f_loc_pert = X_f_loc_pert_tmp.T
            X_f_loc_mean = X_f_loc_mean_tmp.T

            # Local observation
            HX_f_loc_mean = HX_f_mean[d]
            HX_f_loc_pert = np.reshape(HX_f_pert[d,:],(1,N_e))

            # LETKF
            Rinv = np.linalg.inv(np.reshape(self.R[d,d], (1,1)))

            y_loc = obs[d]
            D = y_loc - HX_f_loc_mean # 1 x 1

            A1 = (N_e-1)*np.eye(N_e) 
            A2 = HX_f_loc_pert.T @ Rinv @ HX_f_loc_pert # N_e x N_e 
            A = A1 + A2
            P = np.linalg.inv(A)

            K = np.reshape(X_f_loc_pert @ P @ HX_f_loc_pert.T @ Rinv, N_x_local) # N_x_loc x 1

            X_a_loc_mean = X_f_loc_mean + K * D

            sigma, V = np.linalg.eigh( (N_e - 1) * P )
            X_a_loc_pert = X_f_loc_pert @ V @ np.diag( np.sqrt( np.real(sigma) ) ) @ V.T

            X_a_loc = np.reshape(X_a_loc_mean,(N_x_local,1)) + X_a_loc_pert

            # Calculate weighted local analysis
            weighted_X_a_loc = X_a_loc[:,:]*(np.tile(self.W_loc.flatten().T, (N_e, 1)).T)
            # Here, we use np.tile(W_loc.flatten().T, (N_e_active, 1)).T to repeat W_loc as column vector N_e_active times 

            if not (xroll == 0 and yroll == 0):
                weighted_X_a_loc = np.roll(np.roll(weighted_X_a_loc[:,:].reshape((self.W_loc.shape[0], self.W_loc.shape[1], N_e)), 
                                                                                shift=yroll, axis=0 ), 
                                                shift=xroll, axis=1)


            X_a[:,L] += weighted_X_a_loc.reshape(self.W_loc.shape[0]*self.W_loc.shape[1], N_e).T
        # (end loop over all d)


        # COMBINING (the already weighted) ANALYSIS WITH THE FORECAST
        X_new = np.zeros_like(X_f)
        for e in range(N_e):
            X_new[e] = self.W_forecasts[g]*X_f[e] + X_a[e]

        # Upload
        X_new = np.reshape(X_new, (N_e, nx*ny)).T
        self.statistics.set_ensemble(X_new)

        return X_new


    def filter(self, ensemble, obs):

        for g in range(len(self.groups)):
            ensemble = self.filter_per_group(ensemble, obs[self.groups[g]], g)
    