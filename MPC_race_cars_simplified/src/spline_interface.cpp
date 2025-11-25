#include <iostream>
#include <vector>
#include <cmath>
#include <iostream>
#include <vector>
#include <cmath>
#include "acados_interface.hpp"

int main(int argc, char** argv)
{
    // Simple standalone test that mirrors the Python node core interaction
    const int num_body_points = 6;
    const int N = 50; // horizon length (must match generated model acados_solver_curvilinear_bicycle_model_spatial_N)

    // Example initial compact state: [eY, ePsi, s_body_points..., eY_body_points...]
    std::vector<double> x0(2 + 2 * num_body_points, 0.0);

    // Example body points global positions flattened [x0,x1,...,y0,y1,...]
    std::vector<double> body_points(2 * num_body_points, 0.0);

    // Example spline knots and coeffs placeholder (not used by this demo)
    std::vector<double> spline_knots = {0.0, 1.0};
    std::vector<double> coeffs_x = {0.0, 0.0, 0.0, 0.0};
    std::vector<double> coeffs_y = {0.0, 0.0, 0.0, 0.0};

    // Create wrapper
    AcadosInterface mpc;
    mpc.initialize();

    // Set state (only stage 0)
    mpc.setState(x0.data());

    // Warm start with trivial guesses: (N+1)*NX for x_init and N*NU for u_init
    int NX = CURVILINEAR_BICYCLE_MODEL_SPATIAL_NX;
    int NU = CURVILINEAR_BICYCLE_MODEL_SPATIAL_NU;
    int horizon = CURVILINEAR_BICYCLE_MODEL_SPATIAL_N;

    std::cout << "Number of states (NX): " << NX << std::endl;
    std::cout << "Number of controls (NU): " << NU << std::endl;
    std::cout << "Horizon length (N): " << horizon << std::endl;

    std::vector<double> x_init((horizon+1)*NX, 0.0);
    std::vector<double> u_init(horizon*NU, 0.0);
    mpc.setWarmStart(x_init.data(), u_init.data());

    // Example parameters buffer: use zeros with expected NP
    std::vector<double> p(CURVILINEAR_BICYCLE_MODEL_SPATIAL_NP, 0.0);
    mpc.setParametersAllStages(p.data(), (int)p.size());

    // Solve
    int status = mpc.solve();
    std::cout << "Solver returned status: " << status << std::endl;

    // Get first control
    std::vector<double> u(NU);
    mpc.getControl(u.data());
    std::cout << "First control: ";
    for (int i = 0; i < NU; ++i) std::cout << u[i] << " ";
    std::cout << std::endl;

    // Get full horizon
    std::vector<double> utraj(horizon * NU);
    std::vector<double> xtraj((horizon+1) * NX);
    mpc.getControlTrajectory(utraj.data());
    mpc.getStateTrajectory(xtraj.data());

    std::cout << "\nFull control trajectory (u) \n";
    for (int i = 0; i < horizon; ++i)
    {
        std::cout << "stage " << i << ": ";
        for (int j = 0; j < NU; ++j) std::cout << utraj[i*NU + j] << " ";
        std::cout << std::endl;
    }

    std::cout << "\nFull state trajectory (x) \n";
    for (int i = 0; i <= horizon; ++i)
    {
        std::cout << "stage " << i << ": ";
        for (int j = 0; j < NX; ++j) std::cout << xtraj[i*NX + j] << " ";
        std::cout << std::endl;
    }

    return 0;
}
