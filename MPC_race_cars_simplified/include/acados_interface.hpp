#pragma once

extern "C" {
#include "c_generated_code/acados_solver_curvilinear_bicycle_model_spatial.h"
}
class AcadosInterface {
public:
    AcadosInterface();
    ~AcadosInterface();

    void initialize();
    int solve();
    void setState(const double* x);
    void getControl(double* u) const;
    // Retrieve full horizon of states (length (N+1)*NX)
    void getStateTrajectory(double* xtraj) const;
    // Retrieve full horizon of controls (length N*NU)
    void getControlTrajectory(double* utraj) const;
    // Set parameters for a single stage
    void setParameters(int stage, const double* params, int np);
    // Set the same parameters for all stages
    void setParametersAllStages(const double* params, int np);
    // Warm start: set initial guesses for all states and controls
    void setWarmStart(const double* x_init, const double* u_init);

private:
    curvilinear_bicycle_model_spatial_solver_capsule* capsule_;
};
