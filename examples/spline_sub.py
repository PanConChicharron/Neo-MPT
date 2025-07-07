import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np

class ArraySubscriber(Node):
    def __init__(self):
        super().__init__('array_subscriber')

        self.spline_knots = None
        self.spline_coeffs_x = None
        self.spline_coeffs_y = None
        self.curvatures = None

        self.spline_knots = self.create_subscription(
            Float32MultiArray,
            '/planning/scenario_planning/lane_driving/motion_planning/path_optimizer/debug/spline_knots',
            self.spline_knots_callback,
            10
        )

        self.spline_coeffs_x = self.create_subscription(
            Float32MultiArray,
            '/planning/scenario_planning/lane_driving/motion_planning/path_optimizer/debug/spline_coeffs_x',
            self.spline_coeffs_x_callback,
            10
        )

        self.spline_coeffs_y = self.create_subscription(
            Float32MultiArray,
            '/planning/scenario_planning/lane_driving/motion_planning/path_optimizer/debug/spline_coeffs_y',
            self.spline_coeffs_y_callback,
            10
        )

        self.curvatures = self.create_subscription(
            Float32MultiArray,
            '/planning/scenario_planning/lane_driving/motion_planning/path_optimizer/debug/curvatures',
            self.curvatures_callback,
            10
        )

    def spline_knots_callback(self, msg):
        self.matrix_callback_function(msg, self.spline_knots)

    def spline_coeffs_x_callback(self, msg):
        self.matrix_callback_function(msg, self.spline_coeffs_x)

    def spline_coeffs_y_callback(self, msg):
        self.matrix_callback_function(msg, self.spline_coeffs_y)

    def curvatures_callback(self, msg):
        self.vector_callback_function(msg, self.curvatures)

    def matrix_callback_function(self, msg, variable):
        dims = msg.layout.dim[0]

        if dims.size < 4:
            self.get_logger().warn('Invalid matrix dimensions')
            return

        rows = int(dims.size/dims.stride)
        cols = dims.stride

        flat = np.array(msg.data, dtype=np.float32)
        matrix = flat.reshape((rows, cols))

        variable = matrix
        self.get_logger().info(f"\nReceived 4xN matrix ({msg.layout.dim[0].label}):\n{matrix}")

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
