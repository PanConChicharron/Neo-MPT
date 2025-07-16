import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
plt.ion()
import numpy as np

import sys
import os
import time

from scipy.interpolate import PPoly

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoware_internal_debug_msgs.msg import SplineDebug
from autoware_planning_msgs.msg import Trajectory

from Utils.clothoid_spline import ClothoidSpline
from MPC_race_cars_simplified.path_tracking_mpc_spatial import PathTrackingMPCSpatial

class ArraySubscriber(Node):
    def __init__(self):
        super().__init__('array_subscriber')

        self.spline_knots = None
        self.spline_coeffs_x = None
        self.spline_coeffs_y = None
        self.curvatures = None

        self.N = 100

        self.Sf = 100

        self.spline_knots_callback_obj = self.create_subscription(
            SplineDebug,
            '/planning/scenario_planning/lane_driving/motion_planning/path_optimizer/debug/spline_coefficients',
            self.spline_knots_callback,
            10
        )

        self.optimised_steering_callback_obj = self.create_subscription(
            Float32MultiArray,
            '/planning/scenario_planning/lane_driving/motion_planning/path_optimizer/debug/optimised_steering',
            self.optimised_steering_callback,
            10
        )

        self.optimised_MPT_trajectory_obj = self.create_subscription(
            Trajectory,
            '/planning/scenario_planning/lane_driving/motion_planning/path_optimizer/debug/mpt_traj',
            self.optimised_MPT_trajectory_callback,
            10
        )

        self.path_tracking_mpc_spatial = PathTrackingMPCSpatial(self.Sf, self.N, self.N)

        self.optimised_steering = None
        self.optimised_MPT_trajectory = None

    def optimised_steering_callback(self, msg):
        self.optimised_steering = np.array(msg.data)

    def optimised_MPT_trajectory_callback(self, msg):
        self.optimised_MPT_trajectory = np.array([point for point in msg.points])

    def spline_knots_callback(self, msg):
        t = time.time()
        self.spline_knots = np.array(msg.knots.data)
        # Reverse coefficient order to match SciPy's PPoly expectations
        n_segments = len(self.spline_knots) - 1
        self.spline_coeffs_x = np.array(msg.x_coeffs.data).reshape(4, n_segments)
        self.spline_coeffs_y = np.array(msg.y_coeffs.data).reshape(4, n_segments)
        self.curvatures = np.array(msg.curvatures.data)

        self.clothoid_spline = ClothoidSpline(self.spline_knots[:-1], self.curvatures)

        x0 = np.array([0., 0., 0., 0., 0., 0.])
        simX, simU, Sf, elapsed = self.path_tracking_mpc_spatial.get_optimised_steering(x0, self.clothoid_spline)

        eY=simX[:,0]
        eψ=simX[:,1]
        # Rasterize the spline and plot the x, y spline and the optimized states (eY, eψ)

        # Rasterize the spline
        s_samples = np.linspace(self.spline_knots[0], self.spline_knots[-1], 500)
        x_samples = np.zeros_like(s_samples)
        y_samples = np.zeros_like(s_samples)

        self.spline_x = PPoly(self.spline_coeffs_x, self.spline_knots)
        self.spline_y = PPoly(self.spline_coeffs_y, self.spline_knots)

        # Evaluate the x and y splines using the coefficients
        # Each column in spline_coeffs_x/y is [a3, a2, a1, a0] for a segment
        for i, s in enumerate(s_samples):
            x_samples[i] = self.spline_x(s)
            y_samples[i] = self.spline_y(s)
            
        N = simX.shape[0]
        # 1. Reconstruct s
        s = np.linspace(0, Sf, N)

        # 2. Evaluate reference spline at s
        x_ref = self.spline_x(s)
        y_ref = self.spline_y(s)
        dx_ds = self.spline_x.derivative(1)(s)
        dy_ds = self.spline_y.derivative(1)(s)
        psi_ref = np.arctan2(dy_ds, dx_ds)

        # 3. Transform to global coordinates
        x = x_ref - eY * np.sin(psi_ref)
        y = y_ref + eY * np.cos(psi_ref)
        psi = psi_ref + eψ

        elapsed_time = time.time() - t
        print(f"Time taken for spline processing: {elapsed_time:.4f} seconds")

        # Plot non-blocking and refreshable
        if not hasattr(self, 'fig') or self.fig is None:
            self.fig, self.ax = plt.subplots(1, 2)
        else:
            self.ax[0].clear()
            self.ax[1].clear()

        # Plot trajectory
        self.ax[0].plot(x, y, label='Optimized Trajectory (global)')
        self.ax[0].plot(x_ref, y_ref, '--', label='Reference Spline')
        
        if self.optimised_MPT_trajectory is not None:
            self.ax[0].plot([point.pose.position.x for point in self.optimised_MPT_trajectory], [point.pose.position.y for point in self.optimised_MPT_trajectory], label='MPT Trajectory')
        self.ax[0].set_xlabel('X [m]')
        self.ax[0].set_ylabel('Y [m]')
        self.ax[0].axis('equal')
        self.ax[0].legend()

        #Plot Inputs
        self.ax[1].plot(s, simU, label='Optimized Steering')
        self.ax[1].set_xlabel('s [m]')
        self.ax[1].set_ylabel('delta [rad]')
        # self.ax[1].axis('equal')
        self.ax[1].legend()

        if self.optimised_steering is not None:
            s = np.linspace(0, Sf, len(self.optimised_steering))
            self.ax[1].plot(s, self.optimised_steering, label='Optimized Steering from autoware')
            self.ax[1].set_xlabel('s [m]')
            self.ax[1].set_ylabel('delta [rad]')
            # self.ax[1].axis('equal')
            self.ax[1].legend()


        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        # plt.pause(0.001)

def main(args=None):
    rclpy.init(args=args)
    node = ArraySubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
