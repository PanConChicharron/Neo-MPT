#include "acados_interface.hpp"
#include <cstring>

AcadosInterface::AcadosInterface()
{
    capsule_ = curvilinear_bicycle_model_spatial_acados_create_capsule();
    curvilinear_bicycle_model_spatial_acados_create(capsule_);
}

AcadosInterface::~AcadosInterface()
{
    curvilinear_bicycle_model_spatial_acados_free(capsule_);
    curvilinear_bicycle_model_spatial_acados_free_capsule(capsule_);
}

void AcadosInterface::initialize()
{
    // Example: set initial state to zero
    double x0[CURVILINEAR_BICYCLE_MODEL_SPATIAL_NX] = {0};

    ocp_nlp_in *nlp_in = curvilinear_bicycle_model_spatial_acados_get_nlp_in(capsule_);

    // Set initial state for all shooting nodes
    // use ocp_nlp_out_set to set initial guess for states (and controls below)
    ocp_nlp_config *nlp_config = curvilinear_bicycle_model_spatial_acados_get_nlp_config(capsule_);
    ocp_nlp_dims *nlp_dims = curvilinear_bicycle_model_spatial_acados_get_nlp_dims(capsule_);
    ocp_nlp_out *nlp_out = curvilinear_bicycle_model_spatial_acados_get_nlp_out(capsule_);

    // initialize states and controls to zero
    double u0[CURVILINEAR_BICYCLE_MODEL_SPATIAL_NU] = {0};
    for (int i = 0; i < nlp_dims->N; ++i)
    {
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", x0);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u", u0);
    }
    // final stage x
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, nlp_dims->N, "x", x0);
}

int AcadosInterface::solve()
{
    return curvilinear_bicycle_model_spatial_acados_solve(capsule_);
}

void AcadosInterface::setState(const double *x)
{
    // set current state as initial guess for all stages (or at least stage 0)
    ocp_nlp_config *nlp_config = curvilinear_bicycle_model_spatial_acados_get_nlp_config(capsule_);
    ocp_nlp_dims *nlp_dims = curvilinear_bicycle_model_spatial_acados_get_nlp_dims(capsule_);
    ocp_nlp_in *nlp_in = curvilinear_bicycle_model_spatial_acados_get_nlp_in(capsule_);
    ocp_nlp_out *nlp_out = curvilinear_bicycle_model_spatial_acados_get_nlp_out(capsule_);

    // set as initial guess for stage 0
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, 0, "x", const_cast<double*>(x));
}

void AcadosInterface::getControl(double *u) const
{
    // retrieve control for first stage using ocp_nlp_out_get
    // const_cast used because the generated getters take non-const capsule pointer
    auto cap = const_cast<curvilinear_bicycle_model_spatial_solver_capsule*>(capsule_);
    ocp_nlp_config *nlp_config = curvilinear_bicycle_model_spatial_acados_get_nlp_config(cap);
    ocp_nlp_dims *nlp_dims = curvilinear_bicycle_model_spatial_acados_get_nlp_dims(cap);
    ocp_nlp_out *nlp_out = curvilinear_bicycle_model_spatial_acados_get_nlp_out(cap);

    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 0, "u", u);
}

void AcadosInterface::getStateTrajectory(double* xtraj) const
{
    auto cap = const_cast<curvilinear_bicycle_model_spatial_solver_capsule*>(capsule_);
    ocp_nlp_config *nlp_config = curvilinear_bicycle_model_spatial_acados_get_nlp_config(cap);
    ocp_nlp_dims *nlp_dims = curvilinear_bicycle_model_spatial_acados_get_nlp_dims(cap);
    ocp_nlp_out *nlp_out = curvilinear_bicycle_model_spatial_acados_get_nlp_out(cap);

    int NX = CURVILINEAR_BICYCLE_MODEL_SPATIAL_NX;
    for (int i = 0; i <= nlp_dims->N; ++i)
    {
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, i, "x", &xtraj[i * NX]);
    }
}

void AcadosInterface::getControlTrajectory(double* utraj) const
{
    auto cap = const_cast<curvilinear_bicycle_model_spatial_solver_capsule*>(capsule_);
    ocp_nlp_config *nlp_config = curvilinear_bicycle_model_spatial_acados_get_nlp_config(cap);
    ocp_nlp_dims *nlp_dims = curvilinear_bicycle_model_spatial_acados_get_nlp_dims(cap);
    ocp_nlp_out *nlp_out = curvilinear_bicycle_model_spatial_acados_get_nlp_out(cap);

    int NU = CURVILINEAR_BICYCLE_MODEL_SPATIAL_NU;
    for (int i = 0; i < nlp_dims->N; ++i)
    {
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, i, "u", &utraj[i * NU]);
    }
}

void AcadosInterface::setParameters(int stage, const double *params, int np)
{
    curvilinear_bicycle_model_spatial_acados_update_params(capsule_, stage, const_cast<double *>(params), np);
}

void AcadosInterface::setParametersAllStages(const double *params, int np)
{
    for (int i = 0; i < CURVILINEAR_BICYCLE_MODEL_SPATIAL_N + 1; ++i)
    {
        curvilinear_bicycle_model_spatial_acados_update_params(capsule_, i, const_cast<double *>(params), np);
    }
}

void AcadosInterface::setWarmStart(const double *x_init, const double *u_init)
{
    ocp_nlp_config *nlp_config = curvilinear_bicycle_model_spatial_acados_get_nlp_config(capsule_);
    ocp_nlp_dims *nlp_dims = curvilinear_bicycle_model_spatial_acados_get_nlp_dims(capsule_);
    ocp_nlp_out *nlp_out = curvilinear_bicycle_model_spatial_acados_get_nlp_out(capsule_);
    ocp_nlp_in *nlp_in = curvilinear_bicycle_model_spatial_acados_get_nlp_in(capsule_);

    // x_init is expected to have length (N+1)*NX, u_init length N*NU
    for (int i = 0; i < nlp_dims->N; ++i)
    {
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", const_cast<double*>(&x_init[i * CURVILINEAR_BICYCLE_MODEL_SPATIAL_NX]));
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u", const_cast<double*>(&u_init[i * CURVILINEAR_BICYCLE_MODEL_SPATIAL_NU]));
    }
    // final stage x
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, nlp_dims->N, "x", const_cast<double*>(&x_init[nlp_dims->N * CURVILINEAR_BICYCLE_MODEL_SPATIAL_NX]));
}
