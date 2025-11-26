import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

from scipy.interpolate import PPoly

import rclpy
from geometry_msgs.msg import Quaternion
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoware_internal_debug_msgs.srv import SplineDebug
from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint

from tf_transformations import quaternion_from_euler

from Utils.clothoid_spline import ClothoidSpline
from MPC_race_cars_simplified.path_tracking_mpc_spatial_with_body_points import PathTrackingMPCSpatialWithBodyPoints

class ArraySubscriber(Node):
    def __init__(self):
        super().__init__('array_subscriber')

        self.spline_knots = None
        self.spline_coeffs_x = None
        self.spline_coeffs_y = None
        self.curvatures = None

        self.N = 100

        self.Sf = 100

        self.num_body_points = 6

        self.optimised_trajectory_service = self.create_service(SplineDebug, '/acados_mpt_solver/get_optimised_trajectory', self.get_optimised_trajectory_callback)

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

        # four corners
        self.lf=4.89
        self.lr=0.0
        self.w=1.64
        self.front_overhang=1.0
        self.rear_overhang=1.1
        self.left_overhang=0.128
        self.right_overhang=0.128

        print("Initializing MPC...")
        self.path_tracking_mpc_spatial_with_body_points = PathTrackingMPCSpatialWithBodyPoints(self.Sf, self.N, self.N, self.num_body_points, self.lf, self.lr, self.w, self.front_overhang, self.rear_overhang, self.left_overhang, self.right_overhang)
        print("MPC Initialized.")

        self.optimised_steering = None
        self.optimised_MPT_trajectory = None

    def optimised_steering_callback(self, msg):
        self.optimised_steering = np.array(msg.data)

    def optimised_MPT_trajectory_callback(self, msg):
        self.optimised_MPT_trajectory = np.array([point for point in msg.points])

    def get_optimised_trajectory_callback(self, req, resp):
        spline_knots = np.array(req.knots.data)
        n_segments = len(spline_knots) - 1
        coeffs_x = np.array(req.x_coeffs.data).reshape(4, n_segments)
        coeffs_y = np.array(req.y_coeffs.data).reshape(4, n_segments)
        curvatures = np.array(req.curvatures.data)

        target_segments = self.N
        # import pdb; pdb.set_trace()

        if n_segments < target_segments:
            print(f"Extending from {n_segments} to {target_segments} segments")
            n_missing = target_segments - n_segments - 1
            print(f"n_missing: {n_missing}, n_segments: {n_segments}, target_segments: {target_segments}")
            # Repeat the last knot value
            last_knot = spline_knots[-1]
            ds = self.Sf / self.N
            extra_knots = last_knot + np.arange(1, n_missing+1) * ds
            print(f"extra_knots: {np.shape(extra_knots)}")
            print(f"spline_knots before: {np.shape(spline_knots)}")
            spline_knots = np.concatenate([spline_knots, extra_knots])
            print(f"spline_knots after: {np.shape(spline_knots)}")

            # Linearly extend coefficients: take last segment's end slope
            last_x_coeff = coeffs_x[:, -1].copy()
            last_y_coeff = coeffs_y[:, -1].copy()

            # Create "linear" coefficients — zero out curvature terms
            linear_x_coeff = np.array([0.0, 0.0, last_x_coeff[-2], last_x_coeff[-1]]).reshape(4, 1)
            linear_y_coeff = np.array([0.0, 0.0, last_y_coeff[-2], last_y_coeff[-1]]).reshape(4, 1)

            coeffs_x = np.concatenate([coeffs_x, np.repeat(linear_x_coeff, n_missing, axis=1)], axis=1)
            coeffs_y = np.concatenate([coeffs_y, np.repeat(linear_y_coeff, n_missing, axis=1)], axis=1)

            # Extend curvature as constant
            if (n_missing > 0):
                curvatures = np.concatenate([curvatures, np.repeat(curvatures[-1], n_missing, axis=0)])

        elif n_segments > target_segments:
            print(f"Clipping from {n_segments} to {target_segments} segments")
            # Clip to target segments
            spline_knots = spline_knots[:target_segments]
            coeffs_x = coeffs_x[:, :target_segments-1]
            coeffs_y = coeffs_y[:, :target_segments-1]
            curvatures = curvatures[:target_segments]

        # Save to object
        self.spline_knots = spline_knots
        self.spline_coeffs_x = coeffs_x
        self.spline_coeffs_y = coeffs_y
        self.curvatures = curvatures
        self.body_points_curvilinear = req.body_points_curvilinear
        self.body_points = req.body_points

        # import pdb; pdb.set_trace()

        print("spline_knots: ", np.shape(self.spline_knots))
        print("curvatures: ", np.shape(self.curvatures))

        # self.x_ref_spline = CubicSpline(self.spline_knots[:-1], self.spline_coeffs_x)
        # self.y_ref_spline = CubicSpline(self.spline_knots[:-1], self.spline_coeffs_y)
        self.clothoid_spline = ClothoidSpline(self.spline_knots, self.curvatures)


        x0 = np.array([
            0., 
            0.,
        ])
        
        s_array = [point.x for point in self.body_points_curvilinear]
        eY_array = [point.y for point in self.body_points_curvilinear]
        body_points_curvilinear_array = np.concatenate((s_array, eY_array))

        x = [point.x for point in self.body_points]
        y = [point.y for point in self.body_points]
        body_points_array = np.concatenate((x, y))

        x0 = np.concatenate((x0, body_points_curvilinear_array))
        print(x0)

        t = time.time()
        simX, simU, Sf, elapsed = self.path_tracking_mpc_spatial_with_body_points.get_optimised_steering(x0, body_points_array, self.spline_knots, self.spline_coeffs_x, self.spline_knots, self.spline_coeffs_y, self.clothoid_spline)

        resp.optimised_steering = Float32MultiArray()
        resp.optimised_steering.data = simU.flatten().tolist()

        elapsed_time = time.time() - t
        print(f"Time taken for MPC: {elapsed_time:.4f} seconds")

        t = time.time()

        eY=simX[:,0]
        eψ=simX[:,1]

        s_body_points_N = simX[:, 2 : 2 + self.num_body_points]
        eY_body_points_N = simX[:, 2 + self.num_body_points : ]

        # Rasterize the spline and plot the x, y spline and the optimized states (eY, eψ)

        # Rasterize the spline
        s_samples = np.linspace(self.spline_knots[0], self.spline_knots[-1], 500)
        x_samples = np.zeros_like(s_samples)
        y_samples = np.zeros_like(s_samples)

        # print("s_samples: ", s_samples)
        # print("s_body_points_N: ", s_body_points_N)
        # print("eY_body_points_N: ", eY_body_points_N)

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

        resp.optimised_trajectory = Trajectory()
        for i in range(N):
            point = TrajectoryPoint()
            point.pose.position.x = x[i]
            point.pose.position.y = y[i]

            qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, psi[i])

            q = Quaternion()
            q.x = qx
            q.y = qy
            q.z = qz
            q.w = qw
            point.pose.orientation = q

            resp.optimised_trajectory.points.append(point)

        elapsed_time = time.time() - t
        print(f"Time taken for spline processing: {elapsed_time:.4f} seconds")

        # Plot non-blocking and refreshable
        if not hasattr(self, 'fig') or self.fig is None:
            self.fig, self.ax = plt.subplots(2, 2)
        else:
            self.ax[0, 0].clear()
            self.ax[0, 1].clear()
            self.ax[1, 0].clear()
            self.ax[1, 1].clear()

        # Plot trajectory
        self.ax[0, 0].plot(x, y, label='Optimized Trajectory (global)')
        self.ax[0, 0].plot(x_ref, y_ref, '--', label='Reference Spline')

        # For a straight road, you can simply offset by ±road_width/2 in the normal direction:
        left_x = x_ref - self.path_tracking_mpc_spatial_with_body_points.model.eY_max * np.sin(psi_ref)
        left_y = y_ref + self.path_tracking_mpc_spatial_with_body_points.model.eY_max * np.cos(psi_ref)
        right_x = x_ref - self.path_tracking_mpc_spatial_with_body_points.model.eY_min * np.sin(psi_ref)
        right_y = y_ref + self.path_tracking_mpc_spatial_with_body_points.model.eY_min * np.cos(psi_ref)

        self.ax[0, 0].plot(left_x, left_y, 'k--', alpha=0.5, label='Left Road Boundary')
        self.ax[0, 0].plot(right_x, right_y, 'k--', alpha=0.5, label='Right Road Boundary')

        if self.optimised_MPT_trajectory is None:
            return

        self.ax[0, 0].plot([point.pose.position.x for point in self.optimised_MPT_trajectory], [point.pose.position.y for point in self.optimised_MPT_trajectory], label='MPT Trajectory')

        # Plot the four corners
        for s_body_points, eY_body_points in zip(s_body_points_N, eY_body_points_N):
            x_corners = []
            y_corners = []
            for s_body_point, eY_body_point in zip(s_body_points, eY_body_points):
                dx_ds = self.spline_x.derivative(1)(s_body_point)
                dy_ds = self.spline_y.derivative(1)(s_body_point)
                psi_ref_s_body_point = np.arctan2(dy_ds, dx_ds)
                x_corners.append(self.spline_x(s_body_point) - eY_body_point * np.sin(psi_ref_s_body_point))
                y_corners.append(self.spline_y(s_body_point) + eY_body_point * np.cos(psi_ref_s_body_point))
            
            x_corners.append(x_corners[0])  # Close the rectangle
            y_corners.append(y_corners[0])  # Close the rectangle

            # import pdb; pdb.set_trace()
            
            self.ax[0, 0].plot(x_corners, y_corners)


        self.ax[0, 0].set_xlabel('X [m]')
        self.ax[0, 0].set_ylabel('Y [m]')
        self.ax[0, 0].axis('equal')
        self.ax[0, 0].legend()

        #Plot Inputs
        self.ax[0, 1].plot(s, simU, label='Optimized Steering')
        self.ax[0, 1].set_xlabel('s [m]')
        self.ax[0, 1].set_ylabel('delta [rad]')
        # self.ax[0, 1].axis('equal')
        self.ax[0, 1].legend()

        if self.optimised_steering is None:
            return

        temp_s = np.linspace(0, Sf, len(self.optimised_steering))

        self.ax[0, 1].plot(temp_s, self.optimised_steering, label='Optimized Steering from autoware')
        self.ax[0, 1].set_xlabel('s [m]')
        self.ax[0, 1].set_ylabel('delta [rad]')
        # self.ax[0, 1].axis('equal')
        self.ax[0, 1].legend()

        self.ax[1, 0].plot(s, eY, label='eY')
        self.ax[1, 0].plot(s, eY_body_points_N[:,0], label='eY front left')
        self.ax[1, 0].plot(s, eY_body_points_N[:,1], label='eY front right')
        self.ax[1, 0].plot(s, eY_body_points_N[:,2], label='eY mid right')
        self.ax[1, 0].plot(s, eY_body_points_N[:,3], label='eY rear right')
        self.ax[1, 0].plot(s, eY_body_points_N[:,4], label='eY mid left')
        self.ax[1, 0].plot(s, eY_body_points_N[:,5], label='eY rear left')
        self.ax[1, 0].plot(s, self.path_tracking_mpc_spatial_with_body_points.model.eY_min * np.ones_like(s), '--', label='eY_min')
        self.ax[1, 0].plot(s, self.path_tracking_mpc_spatial_with_body_points.model.eY_max * np.ones_like(s), '--', label='eY_max')
        self.ax[1, 0].set_xlabel('s [m]')
        self.ax[1, 0].legend()


        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        return resp

def main(args=None):
    plt.ion()
    rclpy.init(args=args)
    node = ArraySubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
