from sklearn.gaussian_process import GaussianProcess
import numpy as np
import copy
from matplotlib import pyplot as pl
np.random.seed(1)


def f(x, alpha=5., beta=10.):
    """The function to predict: Weibull centered on 5, ranging from 1 to 2."""
    #return x * np.sin(x)
    return 2. - np.exp(-(x/alpha)**beta)


def smooth_diff(y, smoothing=10):
    """ take a smoothed derivative of y """
    tmp_y = np.array([y[k*smoothing:(k+1)*smoothing].sum()/smoothing for k in xrange((y.shape[0]-1)/smoothing)])
    ret = copy.deepcopy(tmp_y)
    ret[0] = 0
    for k in xrange(tmp_y.shape[0]-1):
        ret[k+1] = tmp_y[k+1] - tmp_y[k]
    return ret


autocorr = 1. # default autocorrelation between points TODO var
target = 1.7 # y value (f(x) value) for which we seek the x TODO var
n_start_points = 2 # number of points that we know for sure
                   # (left and right extremes, thus multiply by 2.)
# now for the extreme points for which we already now the y-values:
# it is very important that they are FAR from the transition zone for two 
# reasons: 1) discrepancy of the transition zone between human subjects
#          2) let/give the GP some freedom/autonomy for a "simpler" regression
# for instance values above 3 for left and below 7 for right give bad results
left_point = 1.5 # known x for which y (f(x)) is 1
right_point = 8.5 # known x for which y (f(x)) is 2
epsilon = 1e-12 # TODO var
n_points = 12 # number of points (on the x axis) that we use TODO 1000
n_samples_per_points = 5 # number of trials/samples for each point TODO var
min_stddev = 0.1
x_meshing = 800 # number of samples on the x axis for plotting/estimating
PLOT = False # do we output the plotting of the GP?
n_estims = 1 # number of estimations for statistical error on x (for which f(x) = target)
if n_estims > 1:
    PLOT = False # do not plot is we are estimating the statistical error

np.random.seed(42)
X = np.ndarray((0,1))
y = np.ndarray((0,1))
dy = np.ndarray((0,1))
sigma = np.array([])
y_pred = np.array([])
# Mesh the input space for evaluations of the real function, 
# the prediction and its MSE
x = np.atleast_2d(np.linspace(1, 9, x_meshing)).T

x_step = (x[-1] - x[0])/x_meshing
s2_x_estim = 0.0
x_estim_values = []
f_x = np.array(map(f, x))
x_found_by_f = np.abs(f_x - target).argmin() * x_step + x[0]
for q in xrange(n_estims):
    # Instanciate a Gaussian Process model
    # theta0 is the autocorrelation parameter
    # thetaL is the lower bound on the autocorrelation parameter
    # thetaU is the upper bound on the autocorrelation parameter
    # nuggets is added to the diag of the training covar (~ Tikhonov regularization)
    gp = GaussianProcess(corr='squared_exponential', theta0=autocorr,
                         #thetaL=1e-2, thetaU=10,
                         random_start=100)
    if n_start_points > 0:
        X = np.array([left_point - 0.01*i for i in xrange(n_start_points)] )
                #+ [right_point + 0.01*i for i in xrange(n_start_points)]) # TODO put back
        X = np.atleast_2d(X).T

        # Observations and noise
        y = f(X).ravel() 
        dy = np.array([epsilon for _ in y])

        # Fit to data using Maximum Likelihood Estimation of the parameters
        print gp.fit(X, y)

        # Make the prediction on the meshed x-axis (ask for MSE as well)
        y_pred, MSE = gp.predict(x, eval_MSE=True)
        sigma = np.sqrt(MSE)

        if PLOT:
            # Plot the function, the prediction and the 95% confidence 
            # interval based on the MSE
            fig = pl.figure()
            pl.plot(x, f(x), 'r:', label=u'$f(x) = 2 - e^{-(x/\\alpha)^\\beta}$')
            pl.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label=u'Observations')
            pl.plot(x, y_pred, 'b-', label=u'Prediction')
            pl.fill(np.concatenate([x, x[::-1]]),
                    np.concatenate([y_pred - 1.9600 * sigma,
                                   (y_pred + 1.9600 * sigma)[::-1]]),
                    alpha=.5, fc='b', ec='None', label='95% confidence interval')
            pl.xlabel('$x$')
            pl.ylabel('$f(x)$')
            pl.ylim(0, 3)
            pl.xlim(1, 9)
            pl.legend(loc='upper left')
            pl.savefig('init.png')

    for i in xrange(n_points): # 10 points time 5 trial per point
        ### new point sample policy
        if sigma.shape[0]:
            ind = sigma.argmax(axis=0)
            #diff = smooth_diff(y_pred, smoothing=10) # TODO try a max derivative policy (weighted by uncertainty)
            #print np.abs(diff).argmax(axis=0)
            #ind = max(0, np.abs(diff).argmax(axis=0)*10 - 5)
        else: # TODO (when we have no points at all at the beginning
            #ind = x.shape[0] / 2.
            ind = 0 
        tmp = x[ind] + np.random.normal(0, 0.0000001)
        print "sampling a new point at x=", tmp
        X = np.append(X, np.array([tmp]), axis=0)

        ### noise due to the experiment's subject input
        values = []
        for j in xrange(n_samples_per_points):
            value = (np.random.uniform(0, 1) > (2 - f(tmp))) + 1 # adds noise
            #value = (np.random.normal(0, 0.1) > (2 - f(tmp))) + 1 # adds noise, TODO try other noise models
            values.append(value)
        mean = sum(values)*1.0 / len(values)
        s2 = 1.0/(max(1.0, len(values)-1)) * sum(map(lambda x: (x-mean)**2, values))
        value = mean
        y = np.append(y, [value])
        dy = np.append(dy, [s2 + min_stddev])

        gp = GaussianProcess(corr='squared_exponential', theta0=autocorr,
                             #thetaL=1e-2, thetaU=10,
                             nugget=(dy/y) ** 2, # regularization
                             random_start=100)

        gp.fit(X, y)
        y_pred, MSE = gp.predict(x, eval_MSE=True)
        sigma = np.sqrt(MSE)

        if PLOT:
            # TODO visualising uncertainty as in http://nbviewer.ipython.org/3947841/bootstrap.ipynb
            fig = pl.figure()
            pl.plot(x, f(x), 'r:', label=u'$f(x) = 2 - e^{-(x/\\alpha)^\\beta}$')
            pl.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label=u'Observations')
            pl.plot(x, y_pred, 'b-', label=u'Prediction')
            pl.fill(np.concatenate([x, x[::-1]]),
                    np.concatenate([y_pred - 1.9600 * sigma,
                                   (y_pred + 1.9600 * sigma)[::-1]]),
                    alpha=.5, fc='b', ec='None', label='95% confidence interval')
            pl.xlabel('$x$')
            pl.ylabel('$f(x)$')
            pl.ylim(0, 3)
            pl.xlim(1, 9)
            pl.legend(loc='upper left')
            pl.savefig(('point_%03d' % i)+'.png')

    x_found_by_GP = np.abs(y_pred - target).argmin() * x_step + x[0]
    s2_x_estim += (x_found_by_GP - x_found_by_f) ** 2
    x_estim_values.append(x_found_by_GP)

print "target (in y):", target
print "x found by f", x_found_by_f
x_mean = sum(x_estim_values)*1.0/len(x_estim_values)
print "mean x estimation:", x_mean
print "sigma^2 on this estimation:", (1.0/(max(1.0, len(x_estim_values)-1))) * sum(map(lambda x: (x-x_mean)**2, x_estim_values))
s2_x_estim /= max(1.0, len(x_estim_values)-1)
print "standard deviation (sigma^2) with x found by f:", s2_x_estim


# And then with ImageMagick:
# convert -delay 50 -loop 0 -dispose background +dither point_*.png movie.gif
#pl.show()


