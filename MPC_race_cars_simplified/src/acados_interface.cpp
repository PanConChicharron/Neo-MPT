#include "acados_interface.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>

#include "c_generated_code/acados_solver_curvilinear_bicycle_model_spatial.h"

AcadosInterface::AcadosInterface()
{
    capsule_ = curvilinear_bicycle_model_spatial_acados_create_capsule();
    curvilinear_bicycle_model_spatial_acados_create(capsule_);

    // Example: set initial state to zero (will be overridden by setInitialState)
    double lbx0[CURVILINEAR_BICYCLE_MODEL_SPATIAL_NX] = {0};
    double ubx0[CURVILINEAR_BICYCLE_MODEL_SPATIAL_NX] = {0};

    nlp_config_ = curvilinear_bicycle_model_spatial_acados_get_nlp_config(capsule_);
    nlp_dims_ = curvilinear_bicycle_model_spatial_acados_get_nlp_dims(capsule_);
    nlp_in_ = curvilinear_bicycle_model_spatial_acados_get_nlp_in(capsule_);
    nlp_out_ = curvilinear_bicycle_model_spatial_acados_get_nlp_out(capsule_);
    nlp_solver_ = curvilinear_bicycle_model_spatial_acados_get_nlp_solver(capsule_);

    ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "ubx", ubx0);
}

AcadosInterface::~AcadosInterface()
{
    curvilinear_bicycle_model_spatial_acados_free(capsule_);
    curvilinear_bicycle_model_spatial_acados_free_capsule(capsule_);
}

void AcadosInterface::setParameters(int stage, std::array<double, NP> params)
{
    curvilinear_bicycle_model_spatial_acados_update_params(capsule_, stage, const_cast<double *>(params.data()), NP);
}

void AcadosInterface::setParametersAllStages(std::array<double, NP> params)
{
    ocp_nlp_dims *nlp_dims = curvilinear_bicycle_model_spatial_acados_get_nlp_dims(capsule_);
    for (int i = 0; i <= nlp_dims->N; ++i)
    {
        curvilinear_bicycle_model_spatial_acados_update_params(capsule_, i, const_cast<double *>(params.data()), NP);
    }
}

void AcadosInterface::setWarmStart(std::array<double, NX> x0, std::array<double, NU> u0)
{
    // Apply warm-start initial guesses to ocp_nlp_out for all stages
    ocp_nlp_dims *nlp_dims = curvilinear_bicycle_model_spatial_acados_get_nlp_dims(capsule_);
    for (int i = 0; i < nlp_dims->N; ++i) {
        ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "x", const_cast<double *>(x0.data()));
        ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "u", const_cast<double *>(u0.data()));
    }
    // final state
    ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, nlp_dims->N, "x", const_cast<double *>(x0.data()));
}

void AcadosInterface::setInitialState(std::array<double, NX> x0, std::array<double, NU> u0)
{
    // set both lbx and ubx at stage 0 to force initial state
    ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "lbx", const_cast<double*>(x0.data()));
    ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "ubx", const_cast<double*>(x0.data()));
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "x", x0.data());
        ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "u", u0.data());
    }
    ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, N, "x", x0.data());
}

std::array<std::array<double, NX>, N+1> AcadosInterface::getStateTrajectory() const
{
    std::array<std::array<double, NX>, N+1> xtraj;
    for (int ii = 0; ii <= nlp_dims_->N; ii++)
    {
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, ii, "x", &xtraj[ii]);
    }
    
    return xtraj;
}

std::array<std::array<double, NU>, N> AcadosInterface::getControlTrajectory() const
{
    std::array<std::array<double, NU>, N> utraj;
    for (int ii = 0; ii < N; ii++)
    {   
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, ii, "u", &utraj[ii]);
    }
    return utraj;
}

int AcadosInterface::solve()
{
    int status = curvilinear_bicycle_model_spatial_acados_solve(capsule_);
    return status;
}

AcadosSolution AcadosInterface::getControl(std::array<double, NX> x0)
{
    // initial value for control input
    std::array<double, NU> u0;
    u0.fill(0.0);

    // prepare evaluation
    int NTIMINGS = 1;
    double min_time = 1e12;
    double kkt_norm_inf;
    double elapsed_time;
    int sqp_iter;

    double xtraj[NX * (N+1)];
    double utraj[NU * N];

    // initialize solution
    setInitialState(x0, u0);
    
    int status = curvilinear_bicycle_model_spatial_acados_solve(capsule_);
    ocp_nlp_get(nlp_solver_, "time_tot", &elapsed_time);
    min_time = std::min(elapsed_time, min_time);

    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 0, "kkt_norm_inf", &kkt_norm_inf);
    ocp_nlp_get(nlp_solver_, "sqp_iter", &sqp_iter);

    curvilinear_bicycle_model_spatial_acados_print_stats(capsule_);
    
    std::stringstream ss;
    ss << "\nSolver info:" << std::endl;
    ss << " SQP iterations " << sqp_iter << "\n minimum time for " << 1 << " solve " << min_time*1000 << " [ms]\n KKT " << kkt_norm_inf << std::endl;

    AcadosSolution solution;
    solution.status = status;
    solution.info = ss.str();
    solution.xtraj = getStateTrajectory();
    solution.utraj = getControlTrajectory();

    return solution;
}