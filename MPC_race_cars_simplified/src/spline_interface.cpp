// ROS2 node that mirrors examples/spline_sub_bicycle_model_spatial_with_body_points.py
// Replace heavy dependencies with lighter approach: include ROS2 headers only if available at build time
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <geometry_msgs/msg/quaternion.hpp>

#include <autoware_internal_debug_msgs/srv/spline_debug.hpp>
#include <autoware_planning_msgs/msg/trajectory.hpp>
#include <autoware_planning_msgs/msg/trajectory_point.hpp>

#include <chrono>
#include <thread>
#include <cassert>
#include <iomanip>

#include "acados_interface.hpp"

using namespace std::chrono_literals;

class SplineInterfaceNode : public rclcpp::Node
{
public:
    using SplineDebug = autoware_internal_debug_msgs::srv::SplineDebug;
    using Trajectory = autoware_planning_msgs::msg::Trajectory;
    using TrajectoryPoint = autoware_planning_msgs::msg::TrajectoryPoint;

    SplineInterfaceNode()
    : Node("spline_interface")
    {
        RCLCPP_INFO(this->get_logger(), "spline_interface node started");

        // subscriptions (store latest messages)
        steering_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/planning/scenario_planning/lane_driving/motion_planning/path_optimizer/debug/optimised_steering",
            10,
            std::bind(&SplineInterfaceNode::steeringCallback, this, std::placeholders::_1)
        );

        mpt_traj_sub_ = this->create_subscription<Trajectory>(
            "/planning/scenario_planning/lane_driving/motion_planning/path_optimizer/debug/mpt_traj",
            10,
            std::bind(&SplineInterfaceNode::mptTrajectoryCallback, this, std::placeholders::_1)
        );

        service_ = this->create_service<SplineDebug>(
            "/acados_mpt_solver/get_optimised_trajectory",
            std::bind(&SplineInterfaceNode::serviceCallback, this, std::placeholders::_1, std::placeholders::_2)
        );

        // Initialize Acados interface
        mpc_.initialize();
    }

private:
    void steeringCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lk(mutex_);
        latest_steering_ = *msg;
    }

    void mptTrajectoryCallback(const autoware_planning_msgs::msg::Trajectory::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lk(mutex_);
        latest_mpt_traj_ = *msg;
    }

    void serviceCallback(const std::shared_ptr<SplineDebug::Request> req, std::shared_ptr<SplineDebug::Response> resp)
    {
        // Parse inputs similar to Python node
        // knots
        std::vector<double> knots(req->knots.data.begin(), req->knots.data.end());
        int n_segments = (int)knots.size() - 1;

        RCLCPP_ERROR(this->get_logger(), "Received request with %d segments", n_segments);

        // x_coeffs and y_coeffs are flattened arrays of length 4 * n_segments
        std::vector<double> x_coeffs_flat(req->x_coeffs.data.begin(), req->x_coeffs.data.end());
        std::vector<double> y_coeffs_flat(req->y_coeffs.data.begin(), req->y_coeffs.data.end());

        // curvatures
        std::vector<double> curvatures(req->curvatures.data.begin(), req->curvatures.data.end());

        int target_segments = CURVILINEAR_BICYCLE_MODEL_SPATIAL_N; // use solver horizon as target

        // Adjust sizes if necessary (simple strategy: extend last values)
        if (n_segments < target_segments) {
            RCLCPP_ERROR(this->get_logger(), "Extending from %d to %d segments", n_segments, target_segments);
            int n_missing = target_segments - n_segments;
            double last_knot = knots.back();
            double ds = 0.0;
            if (knots.size() >= 2) ds = (knots.back() - knots.front()) / (knots.size() - 1);
            for (int i = 0; i < n_missing; ++i) knots.push_back(last_knot + (i+1) * ds);

            // extend coeffs by repeating last column
            for (int i = 0; i < 4; ++i) {
                double v = x_coeffs_flat[(n_segments-1)*4 + i];
                for (int j = 0; j < n_missing; ++j) x_coeffs_flat.push_back(v);
                v = y_coeffs_flat[(n_segments-1)*4 + i];
                for (int j = 0; j < n_missing; ++j) y_coeffs_flat.push_back(v);
            }

            // extend curvatures
            if (!curvatures.empty()) {
                double last_k = curvatures.back();
                for (int i = 0; i < n_missing; ++i) curvatures.push_back(last_k);
            }
        } else if (n_segments > target_segments) {
            RCLCPP_ERROR(this->get_logger(), "Clipping from %d to %d segments", n_segments, target_segments);
            // clip to exactly target_segments segments
            knots.resize(target_segments);
            x_coeffs_flat.resize(4 * (target_segments-1));
            y_coeffs_flat.resize(4 * (target_segments-1));
            curvatures.resize(4*(target_segments-1));
        }

        // Sanity checks based on horizon (target_segments)
        // We expect number of knots = target_segments + 1
        size_t expected_knots = (size_t)target_segments;
        size_t expected_xcoeff = (size_t)4 * (size_t)(target_segments-1); // 4 coefficients per segment, segments == target_segments
        size_t expected_ycoeff = expected_xcoeff;
        size_t expected_curv = (size_t)4 * (size_t)(target_segments-1); // one curvature value per segment

        RCLCPP_ERROR(this->get_logger(), "sizes: knots=%zu x_coeffs=%zu y_coeffs=%zu curvatures=%zu body_points=%zu", knots.size(), x_coeffs_flat.size(), y_coeffs_flat.size(), curvatures.size(), req->body_points.size());

        // Assert basic shape expectations to help catch packing bugs early
        if (knots.size() != expected_knots) {
            RCLCPP_ERROR(this->get_logger(), "unexpected knots length: got=%zu expected=%zu (target_segments=%d)", knots.size(), expected_knots, target_segments);
            assert(knots.size() == expected_knots && "knots length mismatch");
        }
        if (x_coeffs_flat.size() != expected_xcoeff) {
            RCLCPP_ERROR(this->get_logger(), "unexpected x_coeffs length: got=%zu expected=%zu", x_coeffs_flat.size(), expected_xcoeff);
            assert(x_coeffs_flat.size() == expected_xcoeff && "x_coeffs length mismatch");
        }
        if (y_coeffs_flat.size() != expected_ycoeff) {
            RCLCPP_ERROR(this->get_logger(), "unexpected y_coeffs length: got=%zu expected=%zu", y_coeffs_flat.size(), expected_ycoeff);
            assert(y_coeffs_flat.size() == expected_ycoeff && "y_coeffs length mismatch");
        }
        if (curvatures.size() != expected_curv) {
            RCLCPP_ERROR(this->get_logger(), "unexpected curvatures length: got=%zu expected=%zu", curvatures.size(), expected_curv);
            assert(curvatures.size() == expected_curv && "curvatures length mismatch");
        }
        if (req->body_points.size() != req->body_points_curvilinear.size()) {
            RCLCPP_ERROR(this->get_logger(), "body points length mismatch: body_points=%zu body_points_curvilinear=%zu", req->body_points.size(), req->body_points_curvilinear.size());
            assert(req->body_points.size() == req->body_points_curvilinear.size() && "body points mismatch");
        }

        // body points curvilinear -> vector of doubles (s values then eY values)
        std::vector<double> body_points_curvilinear;
        for (const auto &pt : req->body_points_curvilinear) {
            body_points_curvilinear.push_back(pt.x);
        }
        for (const auto &pt : req->body_points_curvilinear) {
            body_points_curvilinear.push_back(pt.y);
        }

        // body points global
        std::vector<double> body_points_xy;
        for (const auto &pt : req->body_points) {
            body_points_xy.push_back(pt.x);
        }
        for (const auto &pt : req->body_points) {
            body_points_xy.push_back(pt.y);
        }

        // Build parameters vector similar to Python: [s_interp, x_ref_sub_knots, x_ref_sub_coeffs_flat, y_ref_sub_knots, y_ref_sub_coeffs_flat, clothoid_sub_knots, clothoid_sub_coeffs_flat, body_points_array]
        // For now, use full knots and coeffs as provided
        std::vector<double> parameters;
        double s_interp = 0.0;
        parameters.push_back(s_interp);

        // x ref knots
        parameters.insert(parameters.end(), knots.begin(), knots.end());
        // x coeffs
        parameters.insert(parameters.end(), x_coeffs_flat.begin(), x_coeffs_flat.end());
        // y ref knots (same as x)
        parameters.insert(parameters.end(), knots.begin(), knots.end());
        // y coeffs
        parameters.insert(parameters.end(), y_coeffs_flat.begin(), y_coeffs_flat.end());

        // clothoid sub: we don't compute a sub-spline here; append curvatures and knots as-is
        parameters.insert(parameters.end(), knots.begin(), knots.end());
        parameters.insert(parameters.end(), curvatures.begin(), curvatures.end());

        // body points
        parameters.insert(parameters.end(), body_points_xy.begin(), body_points_xy.end());

        // set x0: initial state vector
        // minimal x0: [0,0,...body points curvilinear]
        std::vector<double> x0;
        x0.push_back(0.0);
        x0.push_back(0.0);
        // append body_points_curvilinear
        x0.insert(x0.end(), body_points_curvilinear.begin(), body_points_curvilinear.end());

        // Prepare warm start arrays (zeros)
        int NX = CURVILINEAR_BICYCLE_MODEL_SPATIAL_NX;
        int NU = CURVILINEAR_BICYCLE_MODEL_SPATIAL_NU;
        int N = CURVILINEAR_BICYCLE_MODEL_SPATIAL_N;

        std::vector<double> x_init((N+1)*NX, 0.0);
        std::vector<double> u_init(N*NU, 0.0);
        mpc_.setWarmStart(x_init.data(), u_init.data());

        // Set x0 as lbx/ubx by using setState (which sets initial guess for stage 0)
        // If the solver expects constraints on x0 via ocp_nlp_in, our AcadosInterface doesn't expose that, so call setState
        // Fill a full state vector with zeros and copy x0 into its first elements
        std::vector<double> x0_full(NX, 0.0);
        for (size_t i = 0; i < x0.size() && i < (size_t)NX; ++i) x0_full[i] = x0[i];
        mpc_.setState(x0_full.data());

        // set parameters for all stages if NP>0
        if (CURVILINEAR_BICYCLE_MODEL_SPATIAL_NP > 0) {
            size_t actual_np = parameters.size();
            // compute expected from component sizes using the packing used above
            size_t k = knots.size();
            size_t xc = x_coeffs_flat.size();
            size_t yc = y_coeffs_flat.size();
            size_t cv = curvatures.size();
            size_t bp = body_points_xy.size();
            size_t computed_expected = 1 + k + xc + k + yc + k + cv + bp;
            size_t expected_np = (size_t)CURVILINEAR_BICYCLE_MODEL_SPATIAL_NP;

            if (actual_np != expected_np) {
                // Log a small sample to aid debugging
                size_t sample = std::min((size_t)10, actual_np);
                std::string s_first = "";
                for (size_t i = 0; i < sample; ++i) {
                    s_first += std::to_string(parameters[i]) + ", ";
                }
                std::string s_last = "";
                for (size_t i = (actual_np > sample ? actual_np - sample : 0); i < actual_np; ++i) {
                    s_last += std::to_string(parameters[i]) + ", ";
                }
                // Fail fast in debug to help developers catch the issue
                assert(actual_np == expected_np && "parameters length does not match CURVILINEAR_BICYCLE_MODEL_SPATIAL_NP");
            }
            mpc_.setParametersAllStages(parameters.data(), (int)parameters.size());
        }

        int status = mpc_.solve();
        RCLCPP_INFO(this->get_logger(), "Acados solve status: %d", status);

        // Retrieve control trajectory and state trajectory
        std::vector<double> utraj(N * NU);
        std::vector<double> xtraj((N+1) * NX);
        mpc_.getControlTrajectory(utraj.data());
        mpc_.getStateTrajectory(xtraj.data());

        // Fill response: optimised_steering = flattened utraj
        resp->optimised_steering.data.clear();
        for (int i = 0; i < N * NU; ++i) resp->optimised_steering.data.push_back((float)utraj[i]);

        // For trajectory, reconstruct simple points from state trajectory: eY and epsi combined with reference
        // We don't evaluate splines here; provide a minimal trajectory using xtraj states: assume xtraj has columns [eY, epsi, ...]
        int simN = N; // number of points
        resp->optimised_trajectory.points.clear();
        for (int i = 0; i < simN; ++i) {
            autoware_planning_msgs::msg::TrajectoryPoint pt;
            // Set position.x = eY (placeholder) and y = epsi to keep types consistent
            double eY = xtraj[i*NX + 0];
            double epsi = xtraj[i*NX + 1];
            pt.pose.position.x = eY;
            pt.pose.position.y = epsi;
            resp->optimised_trajectory.points.push_back(pt);
        }

        // print steering
        for (size_t i = 0; i < resp->optimised_steering.data.size(); ++i) {
            RCLCPP_INFO(this->get_logger(), "Steering[%zu]: %f", i, resp->optimised_steering.data[i]);
        }

        RCLCPP_INFO(this->get_logger(), "Service handled: produced %zu steering values and %zu trajectory points", resp->optimised_steering.data.size(), resp->optimised_trajectory.points.size());
    }

    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr steering_sub_;
    rclcpp::Subscription<autoware_planning_msgs::msg::Trajectory>::SharedPtr mpt_traj_sub_;
    rclcpp::Service<autoware_internal_debug_msgs::srv::SplineDebug>::SharedPtr service_;

    std::mutex mutex_;
    std_msgs::msg::Float32MultiArray latest_steering_;
    autoware_planning_msgs::msg::Trajectory latest_mpt_traj_;

    AcadosInterface mpc_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SplineInterfaceNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
