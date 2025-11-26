#include "acados_interface.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>

#include "c_generated_code/acados_solver_curvilinear_bicycle_model_spatial.h"

AcadosInterface::AcadosInterface()
{
    capsule_ = curvilinear_bicycle_model_spatial_acados_create_capsule();
    curvilinear_bicycle_model_spatial_acados_create(capsule_);

    // Example: set initial state to zero
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

int AcadosInterface::solve()
{
    int status = curvilinear_bicycle_model_spatial_acados_solve(capsule_);
    return status;
}

int AcadosInterface::getControl(std::array<double, NX> x0) const
{
    // initial value for control input
    double u0[NU];
    u0[0] = 0.0;

    // prepare evaluation
    int NTIMINGS = 1;
    double min_time = 1e12;
    double kkt_norm_inf;
    double elapsed_time;
    int sqp_iter;

    double xtraj[NX * (N+1)];
    double utraj[NU * N];

    // initialize solution
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "x", x0.data());
        ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "u", u0);
    }
    ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, N, "x", x0.data());
    int status = curvilinear_bicycle_model_spatial_acados_solve(capsule_);
    ocp_nlp_get(nlp_solver_, "time_tot", &elapsed_time);
    min_time = std::min(elapsed_time, min_time);

    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 0, "kkt_norm_inf", &kkt_norm_inf);
    ocp_nlp_get(nlp_solver_, "sqp_iter", &sqp_iter);
    
    curvilinear_bicycle_model_spatial_acados_print_stats(capsule_);
    
    std::cerr << "\nSolver info:" << std::endl;
    std::cerr << " SQP iterations " << sqp_iter << "\n minimum time for " << 1 << " solve " << min_time*1000 << " [ms]\n KKT " << kkt_norm_inf << std::endl;

    return status;
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

void AcadosInterface::setParameters(int stage, const double *params, int np)
{
    curvilinear_bicycle_model_spatial_acados_update_params(capsule_, stage, const_cast<double *>(params), np);
}

void AcadosInterface::setParametersAllStages(const double *params, int np)
{
    ocp_nlp_dims *nlp_dims = curvilinear_bicycle_model_spatial_acados_get_nlp_dims(capsule_);
    for (int i = 0; i <= nlp_dims->N; ++i)
    {
        curvilinear_bicycle_model_spatial_acados_update_params(capsule_, i, const_cast<double *>(params), np);
    }
}

void AcadosInterface::setWarmStart(const double *x_init, const double *u_init)
{
    nlp_config_ = curvilinear_bicycle_model_spatial_acados_get_nlp_config(capsule_);
    nlp_dims_ = curvilinear_bicycle_model_spatial_acados_get_nlp_dims(capsule_);
    nlp_out_ = curvilinear_bicycle_model_spatial_acados_get_nlp_out(capsule_);
    nlp_in_ = curvilinear_bicycle_model_spatial_acados_get_nlp_in(capsule_);

    // x_init is expected to have length (N+1)*NX, u_init length N*NU
    for (int i = 0; i < nlp_dims_->N; ++i)
    {
        ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "x", const_cast<double*>(&x_init[i * CURVILINEAR_BICYCLE_MODEL_SPATIAL_NX]));
        ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "u", const_cast<double*>(&u_init[i * CURVILINEAR_BICYCLE_MODEL_SPATIAL_NU]));
    }
    // final stage x
    ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, nlp_dims_->N, "x", const_cast<double*>(&x_init[nlp_dims_->N * CURVILINEAR_BICYCLE_MODEL_SPATIAL_NX]));
}
