import numpy as np
import scipy as sp
#NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)

def estInitialize():
    # Fill in whatever initialization you'd like here. This function generates
    # the internal state of the estimator at time 0. You may do whatever you
    # like here, but you must return something that is in the format as may be
    # used by your estRun() function as the first returned variable.
    #
    # The second returned variable must be a list of student names.
    # 
    # The third return variable must be a string with the estimator type

    """ KALMAN FILTER """
    x = 0.
    y = 0.
    theta = np.pi / 4
    cov = np.diag([25., 25., (np.pi / 6) ** 2])

    internalState = [
        x,
        y,
        theta, 
        cov
    ]

    """ PARTICLE FILTER """
    # num_particles = 10
    # init_particles = np.random.multivariate_normal(
    #     mean=np.zeros(3),
    #     cov=np.diag([25., 25., (np.pi / 6) ** 2]),
    #     size=num_particles
    # )
    # internalState = [
    #     init_particles,
    #     num_particles
    # ]

    # replace these names with yours. Delete the second name if you are working alone.
    studentNames = ['Leonid Pototskiy']
    
    # replace this with the estimator type. Use one of the following options:
    #  'EKF' for Extended Kalman Filter
    #  'UKF' for Unscented Kalman Filter
    #  'PF' for Particle Filter
    #  'OTHER: XXX' if you're using something else, in which case please
    #                 replace "XXX" with a (very short) description
    estimatorType = 'EKF'  
    
    return internalState, studentNames, estimatorType

