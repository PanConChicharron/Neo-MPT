import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utils import CubicSpline1d
from autoware_internal_debug_msgs.msg import SplineDebug

class ArraySubscriber(Node):
    def __init__(self):
        super().__init__('array_subscriber')

        self.spline_knots = None
        self.spline_coeffs_x = None
        self.spline_coeffs_y = None
        self.curvatures = None

        self.spline_knots = self.create_subscription(
            SplineDebug,
            '/planning/scenario_planning/lane_driving/motion_planning/path_optimizer/debug/spline_coefficients',
            self.spline_knots_callback,
            10
        )

    def spline_knots_callback(self, msg):
        self.spline_knots = np.array([knot for knot in msg.knots.data])
        # Reverse coefficient order to match SciPy's PPoly expectations
        n_segments = len(self.spline_knots) - 1
        self.spline_coeffs_x = np.array(msg.x_coeffs.data).reshape(4, n_segments)
        self.spline_coeffs_y = np.array(msg.y_coeffs.data).reshape(4, n_segments)
        self.curvatures = np.array(msg.curvatures.data)

        print(self.spline_knots[-1])

        self.spline_curvatures = CubicSpline1d(self.spline_knots[:-1], self.curvatures)

    def vector_callback_function(self, msg, variable):
        vector = np.array(msg.data, dtype=np.float32)
        self.get_logger().info(f"\nReceived N-length vector:\n{vector}")


def main(args=None):
    rclpy.init(args=args)
    node = ArraySubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
