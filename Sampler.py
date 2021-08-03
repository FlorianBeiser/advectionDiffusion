import numpy as np

class Sampler:
    def __init__(self, grid, args):
        self.grid = grid

        self.args = args

        self.construct()


    def construct(self):

        # Mean: constant lift
        mean_lift = self.args["mean_upshift"]*np.ones(self.grid.N_x)

        if "bell_scaling" in self.args.keys():
            # Mean: bell shape
            xx, yy = np.meshgrid(np.arange(self.grid.nx)*self.grid.dx, np.arange(self.grid.ny)*self.grid.dy)

            bell_scaling = self.args["bell_scaling"]
            bell_sharpness = self.args["bell_sharpness"]

            bell_center_x = self.args["bell_center"][0]*self.grid.xdim
            bell_center_y = self.args["bell_center"][1]*self.grid.ydim

            bell = bell_scaling * np.exp(-bell_sharpness*((xx-bell_center_x)**2 + (yy-bell_center_y)**2))

        else:
            bell = np.zeros((self.grid.ny, self.grid.nx))

        self.mean = mean_lift + np.reshape(bell, self.grid.N_x)
        ######################

        # Cov: Matern like matrix
        phi = self.args["matern_phi"]
        dist_mat = np.copy(self.grid.dist_mat)
        self.cov = (1+phi*dist_mat)*np.exp(-phi*dist_mat)

        #######################

        # Var in each grid node
        var_mesh = self.args["variance"]*np.ones((self.grid.ny, self.grid.nx))
        self.var = np.reshape(var_mesh, self.grid.N_x)


    def sample(self, N=1):
        
        sample = self.gaussian_random_fieldFFT(self.mean, self.cov, self.var, N)

        return sample


    def gaussian_random_field(self, mean, cov, var, N=1):

        sample = np.random.multivariate_normal(mean, cov + 0.01*np.eye(self.grid.N_x), N).transpose()
        sample = np.reshape(var, (self.grid.N_x,1)) * sample

        return sample


    def gaussian_random_fieldFFT(self, mean, cov, var, N=1):

        sample = np.zeros((self.grid.N_x, N))

        # Sampling Gaussian random fields using the FFT
        # What is utilizing the Toepitz structure of the covariance matrix.
        # In the end, it is transformed with the mean and point variances
        # NOTE: For periodic boundary conditions the covariance matrix 
        # becomes numerical problems with the semi-positive definiteness
        # what forbids to use classical np.random.multivariate_normal sampling
        # but the FFT approach for Toepitz matrixes circumvents those problems
        cov_toepitz = np.reshape(cov[0,:], (self.grid.ny, self.grid.nx))
        cmf = np.real(np.fft.fft2(cov_toepitz))

        for e in range(N):
            u = np.random.normal(size=(self.grid.ny, self.grid.nx))
            uif = np.fft.ifft2(u)

            xf = np.real(np.fft.fft2(np.sqrt(np.maximum(cmf,0))*uif))

            sample[:,e] = mean + np.reshape(var, self.grid.N_x)*np.reshape(xf, self.grid.N_x)

        return sample
