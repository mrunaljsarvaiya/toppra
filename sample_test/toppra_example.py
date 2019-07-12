#!/usr/bin/env python

# Toppra imports
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
import copy

def sample_traj(jnt_traj, freq = 1):
    # Sample the trajectory at a given frequency
    # ts_sample = duration*frequency.
    ts_sample = np.linspace(0, jnt_traj.get_duration(), int(jnt_traj.get_duration()*freq))
    qds_sample = jnt_traj.evald(ts_sample)
    qdds_sample = jnt_traj.evaldd(ts_sample)

    return (qds_sample, qdds_sample)

if __name__ == "__main__":

    # Load data
    # Waypts is a 2d numpy array of size 384 by 7.
    # 384 points in trajectory, ndof = 7
    waypts = np.load('toppra_test.npy')
    n = len(waypts)
    ndof = len(waypts[0])

    # Parametrizes the initial path
    path = ta.SplineInterpolator(np.linspace(0, 1, n), waypts)

    # Create velocity and acceleration bounds. Supposing symmetrical bounds around zero.
    vlim_ = np.array([0.87, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87])
    alim_ = np.array([0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0])
   
    vlim = np.vstack((-vlim_, vlim_)).T
    alim = np.vstack((-alim_, alim_)).T
    pc_vel = constraint.JointVelocityConstraint(vlim)
    pc_acc = constraint.JointAccelerationConstraint(
        alim, discretization_scheme=constraint.DiscretizationType.Interpolation)

    # Setup a parametrization instance
    gridpoints = np.linspace(0, path.duration, n*2)
    instance = algo.TOPPRA([pc_vel, pc_acc], path, solver_wrapper='seidel', gridpoints = gridpoints)

    # Compute and sample trajectory
    jnt_traj, _ = instance.compute_trajectory(0, 0)
    (vel, acc) = sample_traj(jnt_traj, freq = 40)
    
    # Get max values
    # Creating a deepcopy to not modify orignal data
    # Taking the absolute values of all points since the acceleration and velocity limits are symetric
    # Expect the max_vel[i] < vlim_[i], max_acc[i] < alim_[i]

    max_vel = ndof*[0]
    max_acc = ndof*[0]
    vel_abs = copy.deepcopy(vel)
    acc_abs = copy.deepcopy(acc)
    vel_abs = np.absolute(vel_abs)
    acc_abs = np.absolute(acc_abs)
    
    for i in range(0, ndof):
        max_vel[i] = np.max(vel_abs[:,i])
        max_acc[i] = np.max(acc_abs[:,i])
    
    for i in range(0, ndof):
        print("Joint {}: \nVel limit: {}, Acc limit: {}".format(i+1, vlim_[i], alim_[i]))
        print("Max Vel: {:.2f}, Max Acc: {:.2f}\n".format(max_vel[i], max_acc[i]))
