#pragma once

#include <array>
#include <stddef.h>
extern "C" {
#include "c_generated_code/acados_solver_curvilinear_bicycle_model_spatial.h"
}

#define NX     CURVILINEAR_BICYCLE_MODEL_SPATIAL_NX
#define NP     CURVILINEAR_BICYCLE_MODEL_SPATIAL_NP
#define NU     CURVILINEAR_BICYCLE_MODEL_SPATIAL_NU
constexpr size_t N      = CURVILINEAR_BICYCLE_MODEL_SPATIAL_N;

class AcadosInterface {
public:
    AcadosInterface();
    ~AcadosInterface();

    int solve();
    int getControl(std::array<double, NX> x0) const;
    // Retrieve full horizon of states (length (N+1)*NX)
    std::array<std::array<double, NX>, N+1> getStateTrajectory() const;
    // Retrieve full horizon of controls (length N*NU)
    std::array<std::array<double, NU>, N> getControlTrajectory() const;
    // Set parameters for a single stage
    void setParameters(int stage, const double* params, int np);
    // Set the same parameters for all stages
    void setParametersAllStages(const double* params, int np);
    // Warm start: set initial guesses for all states and controls
    void setWarmStart(const double* x_init, const double* u_init);

private:
    curvilinear_bicycle_model_spatial_solver_capsule* capsule_;
    ocp_nlp_config *nlp_config_;
    ocp_nlp_dims *nlp_dims_;
    ocp_nlp_in *nlp_in_;
    ocp_nlp_out *nlp_out_;
    ocp_nlp_solver *nlp_solver_;
};
