import numpy as np
from scipy.stats import multivariate_normal
#NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)

""" NOISE ESTIMATION

# Use calibration file
data = np.genfromtxt('../data/run_000.csv', delimiter=',')
# Extract x and y measurements
data = data[:, 3:5]
# Remove nan values
data = data[~np.isnan(data[:, 0]), :]
# Compute covariance matrix
meas_cov = np.cov(data.T)

""" # RESULT:
meas_cov = np.matrix([
    [1.08933973, 1.53329122],
    [1.53329122, 2.98795486]
])

# Model constants
r = .425
B = .8
# The actual state only has dimension 3, but we also want to pass covariance
# Measurement has dimension 2
n = 3
m = 2

x_sd = 4e-1 * .1
y_sd = 4e-1 * .1
theta_sd = np.pi / 3 * .1
process_cov = np.diag([x_sd ** 2, y_sd ** 2, theta_sd ** 2])

def EKF(time, dt, internalStateIn, steeringAngle, pedalSpeed, measurement):
    x = internalStateIn[0]
    y = internalStateIn[1]
    theta = internalStateIn[2]
    cov = internalStateIn[3]

    """ PREDICTION STEP """
    wheel_speed = 5 * pedalSpeed * r
    # Linearization matrices
    A = np.matrix([
        [1., 0., -wheel_speed * np.sin(theta) * dt],
        [0., 1., wheel_speed * np.cos(theta) * dt],
        [0., 0., 1.]
    ])

    # Push estimate through process model
    pred_state = np.matrix([
        [x + wheel_speed * np.cos(theta) * dt],
        [y + wheel_speed * np.sin(theta) * dt],
        [theta + wheel_speed / B * np.tan(steeringAngle) * dt]
    ])

    # Predicted covariance
    pred_cov = A @ cov @ A.T + process_cov

    """CORRECTION STEP"""
    if not (np.isnan(measurement[0]) or np.isnan(measurement[1])):
        meas = np.matrix([[measurement[0]], [measurement[1]]])
        pred_x = pred_state[0, 0]
        pred_y = pred_state[1, 0]
        pred_theta = pred_state[2, 0]

        # Linearize measurement around predicted state
        H = np.matrix([
            [1., 0., -.5 * B * np.sin(pred_theta)],
            [0., 1., .5 * B * np.cos(pred_theta)]
        ])

        # Predicted measurement
        pred_meas = np.matrix([
            [pred_x + .5 * B * np.cos(pred_theta)],
            [pred_y + .5 * B * np.sin(pred_theta)]
        ])

        # Compute Kalman gain and measurement update
        kalman_gain = pred_cov @ H.T @ np.linalg.inv(H @ pred_cov @ H.T + meas_cov)
        corr_state = pred_state + kalman_gain @ (meas - pred_meas)
        inter_mat = np.eye(n) - kalman_gain @ H
        x = corr_state[0, 0]
        y = corr_state[1, 0]
        theta = corr_state[2, 0]
        cov = inter_mat @ pred_cov @ inter_mat.T + kalman_gain @ meas_cov @ kalman_gain.T
    else:
        x = pred_state[0, 0]
        y = pred_state[1, 0]
        theta = pred_state[2, 0]
        cov = pred_cov
    
    internalStateOut = [
        x,
        y,
        theta,
        cov
    ]

    return x, y, theta, internalStateOut


def UKF(time, dt, internalStateIn, steeringAngle, pedalSpeed, measurement):
    x = internalStateIn[0]
    y = internalStateIn[1]
    theta = internalStateIn[2]
    cov = internalStateIn[3]

    """ PREDICTION STEP """
    wheel_speed = 5 * pedalSpeed * r
    state = [x, y, theta]

    # Compute sigma points
    sigma_points = np.zeros((n, 2*n))
    cov_sqrt = np.linalg.cholesky(n * cov)
    for i in range(n):
        sigma_points[:, i] = state + cov_sqrt[:, i]
        sigma_points[:, n+i] = state - cov_sqrt[:, i]

    # Push sigma points through process model
    pred_sigma_points = np.zeros((n, 2*n))
    for i in range(2*n):
        sx, sy, stheta = sigma_points[:, i]
        pred_sigma_points[:, i] = [
            sx + wheel_speed * np.cos(stheta) * dt,
            sy + wheel_speed * np.sin(stheta) * dt,
            stheta + wheel_speed / B * np.tan(steeringAngle) * dt
        ]

    # Compute prior statistics
    pred_state = np.mean(pred_sigma_points, axis=1)
    pred_cov = np.array(process_cov)
    for i in range(2*n):
        diff = pred_sigma_points[:, i] - pred_state
        pred_cov += np.outer(diff, diff) / (2 * n)
    
    """ CORRECTION STEP """
    if not (np.isnan(measurement[0]) or np.isnan(measurement[1])):
        # Push prior sigma points through measurement model
        meas_sigma_points = np.zeros((m, 2*n))
        for i in range(2*n):
            sx, sy, stheta = pred_sigma_points[:, i]
            meas_sigma_points[:, i] = [
                sx + .5 * B * np.cos(stheta),
                sy + .5 * B * np.sin(stheta)
            ]
        
        # Compute posterior statistics
        pred_meas = np.mean(meas_sigma_points, axis=1)
        Pzz = np.array(meas_cov)
        Pxz = np.zeros((n, m))
        for i in range(2*n):
            meas_diff = meas_sigma_points[:, i] - pred_meas
            state_diff = pred_sigma_points[:, i] - pred_state
            Pzz += np.outer(meas_diff, meas_diff) / (2 * n)
            Pxz += np.outer(state_diff, meas_diff) / (2 * n)

        # Kalman gain and measurement update
        kalman_gain = Pxz @ np.linalg.inv(Pzz)
        x, y, theta = pred_state + kalman_gain @ (measurement - pred_meas)
        cov = pred_cov - kalman_gain @ Pzz @ kalman_gain.T
    else:
        x, y, theta = pred_state
        cov = pred_cov
    
    internalStateOut = [
        x,
        y,
        theta,
        cov
    ]

    return x, y, theta, internalStateOut


def PF(time, dt, internalStateIn, steeringAngle, pedalSpeed, measurement):
    particles = internalStateIn[0]
    num_particles = internalStateIn[1]

    """ PREDICTION STEP """
    wheel_speed = 5 * pedalSpeed * r
    # Sample noise and push particles through process model
    process_noise = np.random.multivariate_normal(mean=np.zeros(3), cov=process_cov, size=num_particles)
    pred_particles = np.zeros((num_particles, n))
    for i in range(num_particles):
        xm, ym, thetam = particles[i, :]
        pred_particles[i, :] = [
            xm + wheel_speed * np.cos(thetam) * dt + process_noise[i, 0],
            ym + wheel_speed * np.sin(thetam) * dt + process_noise[i, 1],
            thetam + wheel_speed / B * np.tan(steeringAngle) * dt + process_noise[i, 2]
        ]

    """ CORRECTION STEP """
    if not (np.isnan(measurement[0]) or np.isnan(measurement[1])):
        # Compute weights (measurement likelihood)
        meas_rv = multivariate_normal(mean=np.zeros(m), cov=meas_cov)
        weights = np.zeros(num_particles)
        for i in range(num_particles):
            xp, yp, thetap = pred_particles[i, :]
            weights[i] = meas_rv.pdf([
                measurement[0] - (xp + .5 * B * np.cos(thetap)),
                measurement[1] - (yp + .5 * B * np.cos(thetap))
            ])
        weights /= np.sum(weights)

        # Resample particles
        for i in range(num_particles):
            prob_sample = np.random.uniform()
            sample_idx = 0
            weight_sum = 0

            while weight_sum + weights[sample_idx] < prob_sample:
                weight_sum += weights[sample_idx]
                sample_idx += 1

            particles[i, :] = pred_particles[sample_idx, :]
        
        # Final estimate is mean of particles
        x, y, theta = np.mean(particles, axis=0)
    else:
        x, y, theta = np.mean(pred_particles, axis=0)
        particles = pred_particles
    
    internalStateOut = [
        particles,
        num_particles
    ]

    return x, y, theta, internalStateOut