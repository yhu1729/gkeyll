#include <gkyl_gyrokinetic_priv.h>

#ifdef GKYL_HAVE_SUNDIALS

#include <gkyl_nvector_array.h>

#include <arkode/arkode.h>
#include <arkode/arkode_lsrkstep.h>
#include <arkode/arkode_mristep.h>

// Time stepper that uses the extended super time stepping (ExtSTS) methods in
// SUNDIALS (ARKODE). The state vector wraps the distribution functions of the
// evolving species, so SUNDIALS advances them in place. It also holds the
// running time integral of the stepped (boundary-flux) moments, from which the
// average of those moments over each step is obtained.
struct gk_extsts {
  SUNContext sunctx; // SUNDIALS context.
  N_Vector state; // State vector.
  void *arkode_mem; // ExtSTS (MRIStep) integrator.
  void *sts_mem; // Super time stepping (LSRKStep) integrator, owned by arkode_mem.

  int num_arrays; // Number of arrays in the state vector.
  int *f_idx; // Index in the state of the distribution of each species (-1 if static).
  int *smom_idx; // Index in the state of the first stepped moment of each species.
  struct gkyl_range empty_range; // Excludes the stepped moments from norms.
  struct gkyl_array **f_state; // Distribution of each species in the vector being evaluated.
  struct gkyl_array ***smom_rate; // Stepped-moment rates of each species in the rhs vector.

  double dom_eig_fac; // Dominant eigenvalue / LBO CFL frequency.
  double dt_expl; // Stable time step of the latest explicit rhs.
  double dt_expl_min; // Smallest stable time step of the explicit rhs in this step attempt.
  double omega_diff; // LBO CFL frequency of the latest diffusive rhs on this rank.

  bool adaptive; // Whether ARKODE's temporal error control chooses the time step.
  double rtol; // Adaptive: relative tolerance.
  double atol_frac; // Adaptive: absolute tolerance / largest zeroth DG coefficient of f.
  double *atol; // Adaptive: absolute tolerance of each array of the state.
  double dt_next; // Adaptive: time step ARKODE attempts next.

  // Prepare ARKODE for a step from tcurr of at most dt0. Sets the expected time
  // step (dt) and the output time to pass to ARKodeEvolve (tout).
  int (*step_begin)(
    gkyl_gyrokinetic_app *app, double tcurr, double dt0, double *dt, double *tout
  );
  // Finish a step of dt_actual that was expected to be dt. Returns the suggested next step.
  double (*step_end)(gkyl_gyrokinetic_app *app, double dt, double dt_actual);

  long num_steps; // Number of time steps.
  long num_rhs_expl, num_rhs_diff; // Number of explicit and diffusive rhs evaluations.
  long num_cfl_exceeded; // Steps in which a stage had a smaller stable step than the one taken.
};

// Point the rhs arguments of the app at the arrays of the input and output vectors.
static void
extsts_set_fdot_args(gkyl_gyrokinetic_app *app, N_Vector y, N_Vector ydot)
{
  struct gk_extsts *ext = app->extsts;
  struct gkyl_gyrokinetic_fdot_args *args = &app->fdot_args;

  for (int i = 0; i < app->num_species; ++i) {
    struct gk_species *gks = &app->species[i];
    args->smom_in[i] = gks->step_moms.f;
    if (ext->f_idx[i] < 0) {
      // Static species: not part of the state, and its rhs methods write nothing.
      ext->f_state[i] = gks->f;
      args->fout[i] = gks->f1;
      args->smom_out[i] = gks->step_moms.f1;
    } else {
      // Only the ghost cells of the input are modified (by the BCs).
      ext->f_state[i] = gkyl_nvector_array_get(y, ext->f_idx[i]);
      args->fout[i] = gkyl_nvector_array_get(ydot, ext->f_idx[i]);
      for (int m = 0; m < gks->step_moms.num; ++m) {
        ext->smom_rate[i][m] = gkyl_nvector_array_get(ydot, ext->smom_idx[i] + m);
        gkyl_array_clear(ext->smom_rate[i][m], 0.0);
      }
      args->smom_out[i] = ext->smom_rate[i];
    }
    args->fin[i] = ext->f_state[i];
  }
}

// Time rate of change due to every term except the LBO collisions.
static int
extsts_rhs_explicit(sunrealtype tcurr, N_Vector y, N_Vector ydot, void *user_data)
{
  gkyl_gyrokinetic_app *app = user_data;
  struct gk_extsts *ext = app->extsts;
  struct gkyl_gyrokinetic_fdot_args *args = &app->fdot_args;
  const struct gkyl_array **fin = args->fin;
  const struct gkyl_array **fin_neut = args->fin_neut;
  struct gkyl_array **fout = args->fout;
  struct gkyl_array ***smom_out = args->smom_out;

  extsts_set_fdot_args(app, y, ydot);

  gyrokinetic_calc_field_and_apply_bc(app, tcurr, ext->f_state, args->smom_in, NULL);

  for (int i = 0; i < app->num_species; ++i) {
    struct gk_species *gks = &app->species[i];
    gk_species_bgk_moms(app, gks, &gks->bgk, fin[i]);
  }

  for (int i = 0; i < app->num_species; ++i) {
    struct gk_species *gks = &app->species[i];
    gk_species_bgk_cross_moms(app, gks, &gks->bgk, fin[i]);
    gk_species_react_cross_moms(app, gks, &gks->react, fin, fin_neut);
    gk_species_react_cross_moms(app, gks, &gks->react_neut, fin, fin_neut);
    gk_species_radiation_moms(app, gks, &gks->rad, fin, fin_neut);
    gk_species_scaling_cross_moms(app, gks, &gks->sca, fin, fin_neut);
  }

  // Compute df/dt (not including sources nor the LBO collisions).
  double dt_local = DBL_MAX;
  for (int i = 0; i < app->num_species; ++i) {
    double dt_species =
      gk_species_rhs_explicit(app, &app->species[i], fin[i], fout[i], smom_out[i]);
    dt_local = fmin(dt_local, dt_species);
  }

  // Sources. Done after df/dt in case boundary fluxes are needed.
  for (int i = 0; i < app->num_species; ++i) {
    gk_species_source_rhs(app, &app->species[i], &app->species[i].src, fin[i], fout[i]);
  }

  for (int i = 0; i < app->num_species; ++i) {
    struct gk_species *gks = &app->species[i];
    gk_species_fdot_multiplier_advance_times_rate(
      app, gks, &gks->fdot_mult, app->field->phi_smooth, fin[i], fout[i]
    );
  }

  struct timespec wtm = gkyl_wall_clock();
  double dt_global = dt_local;
  gkyl_comm_allreduce_host(app->comm, GKYL_DOUBLE, GKYL_MIN, 1, &dt_local, &dt_global);
  app->stat.dfdt_dt_reduce_tm += gkyl_time_diff_now_sec(wtm);

  ext->dt_expl = dt_global;
  ext->dt_expl_min = fmin(ext->dt_expl_min, dt_global);
  ext->num_rhs_expl += 1;
  return 0;
}

// Time rate of change due to the LBO collisions.
static int
extsts_rhs_diffusive(sunrealtype tcurr, N_Vector y, N_Vector ydot, void *user_data)
{
  gkyl_gyrokinetic_app *app = user_data;
  struct gk_extsts *ext = app->extsts;
  struct gkyl_gyrokinetic_fdot_args *args = &app->fdot_args;
  const struct gkyl_array **fin = args->fin;
  struct gkyl_array **fout = args->fout;

  // The collisions are local in configuration space: no fields or BCs needed.
  extsts_set_fdot_args(app, y, ydot);

  for (int i = 0; i < app->num_species; ++i) {
    struct gk_species *gks = &app->species[i];
    gk_species_lbo_moms(app, gks, &gks->lbo, fin[i]);
  }
  for (int i = 0; i < app->num_species; ++i) {
    struct gk_species *gks = &app->species[i];
    gk_species_lbo_cross_moms(app, gks, &gks->lbo, fin[i]);
  }

  double omega_local = 0.0;
  for (int i = 0; i < app->num_species; ++i) {
    struct gk_species *gks = &app->species[i];
    double omega_species = gk_species_rhs_diffusive(app, gks, fin[i], fout[i]);
    omega_local = fmax(omega_local, omega_species);

    gk_species_fdot_multiplier_advance_times_rate(
      app, gks, &gks->fdot_mult, app->field->phi_smooth, fin[i], fout[i]
    );
  }

  ext->omega_diff = omega_local;
  ext->num_rhs_diff += 1;
  return 0;
}

// Dominant eigenvalue of the LBO collisions, estimated from the CFL frequency
// of the latest diffusive rhs (which SUNDIALS evaluates right before this).
static int
extsts_dom_eig(
  sunrealtype tcurr, N_Vector y, N_Vector fn, sunrealtype *lambda_real, sunrealtype *lambda_imag,
  void *user_data, N_Vector temp1, N_Vector temp2, N_Vector temp3
)
{
  gkyl_gyrokinetic_app *app = user_data;
  struct gk_extsts *ext = app->extsts;

  double omega_global = ext->omega_diff;
  gkyl_comm_allreduce_host(app->comm, GKYL_DOUBLE, GKYL_MAX, 1, &ext->omega_diff, &omega_global);

  *lambda_real = -ext->dom_eig_fac * omega_global;
  *lambda_imag = 0.0;
  return 0;
}

// Called before each step attempt (ARKODE may reject an attempt and retry).
static int
extsts_pre_step(sunrealtype tcurr, N_Vector y, long int step, int attempt, void *user_data)
{
  gkyl_gyrokinetic_app *app = user_data;
  app->extsts->dt_expl_min = DBL_MAX;
  return 0;
}

// Adaptive: error weights 1/(rtol*|y| + atol), with an absolute tolerance for
// each array of the state.
static int
extsts_ewt(N_Vector y, N_Vector ewt, void *user_data)
{
  gkyl_gyrokinetic_app *app = user_data;
  struct gk_extsts *ext = app->extsts;

  N_VAbs(y, ewt);
  N_VScale(ext->rtol, ewt, ewt);
  for (int i = 0; i < ext->num_arrays; ++i) {
    struct gkyl_array *ewt_arr = gkyl_nvector_array_get(ewt, i);
    for (unsigned k = 0; k < ewt_arr->ncomp; ++k) {
      gkyl_array_shiftc(ewt_arr, ext->atol[i], k);
    }
  }
  N_VInv(ewt, ewt);
  return 0;
}

// Adaptive: the stable time step of the explicit terms from their latest
// evaluation (already reduced over the communicator) caps the time step.
static int
extsts_stable_dt(N_Vector y, sunrealtype tcurr, sunrealtype *hstab, void *user_data)
{
  gkyl_gyrokinetic_app *app = user_data;
  *hstab = app->extsts->dt_expl;
  return 0;
}

// Fixed time step: the stable time step of the explicit terms from their latest
// evaluation, but at most dt0.
static int
extsts_step_begin_fixed(
  gkyl_gyrokinetic_app *app, double tcurr, double dt0, double *dt, double *tout
)
{
  struct gk_extsts *ext = app->extsts;
  *dt = fmin(dt0, ext->dt_expl);
  *tout = tcurr + *dt;
  return ARKodeSetFixedStep(ext->arkode_mem, *dt);
}

static double
extsts_step_end_fixed(gkyl_gyrokinetic_app *app, double dt, double dt_actual)
{
  return app->extsts->dt_expl;
}

// Adaptive time step: ARKODE chooses it, without stepping past tcurr+dt0. The
// expected step is the one ARKODE plans to attempt, but it may reject that
// attempt and take a smaller step.
static int
extsts_step_begin_adaptive(
  gkyl_gyrokinetic_app *app, double tcurr, double dt0, double *dt, double *tout
)
{
  struct gk_extsts *ext = app->extsts;
  *dt = fmin(dt0, ext->dt_next);
  *tout = tcurr + dt0;
  // Only set a stop time when dt0 binds: after a step that ends at the stop
  // time, ARKODE reports the step just taken as the next one (not the one its
  // controller chose), and feeding that back as dt0 would keep the step from growing.
  if (dt0 < ext->dt_next) {
    return ARKodeSetStopTime(ext->arkode_mem, *tout);
  }
  return ARKodeClearStopTime(ext->arkode_mem);
}

static double
extsts_step_end_adaptive(gkyl_gyrokinetic_app *app, double dt, double dt_actual)
{
  struct gk_extsts *ext = app->extsts;

  // The time-rate diagnostics stored at the beginning of the step were divided
  // by the expected time step: rescale them if the step taken differs.
  if (dt_actual != dt) {
    for (int i = 0; i < app->num_species; ++i) {
      struct gk_species *gks = &app->species[i];
      gk_species_scale_integ_mom_dt(app, gks, dt / dt_actual, gks->fdot_mom_old);
    }
    gk_field_scale_energy_dt_old(app, app->field, dt / dt_actual);
  }

  sunrealtype dt_next = 0.0;
  int status = ARKodeGetCurrentStep(ext->arkode_mem, &dt_next);
  if (status != ARK_SUCCESS || !(dt_next > 0.0)) {
    // Not expected after a successful step; fall back to the step just taken.
    dt_next = dt_actual;
  }
  ext->dt_next = dt_next;
  return dt_next;
}

static void
extsts_check(const gkyl_gyrokinetic_app *app, int status, const char *call)
{
  if (status < 0) {
    gkyl_gyrokinetic_app_cout(app, stderr, "ExtSTS: %s failed with status %d.\n", call, status);
    gkyl_exit("ExtSTS time stepper setup failed.");
  }
}

// Adaptive: the absolute tolerance of the distribution of each evolving species
// is atol_frac times the largest magnitude of its zeroth DG coefficient
// (proportional to the cell average) over all MPI ranks.
static void
extsts_set_atol(gkyl_gyrokinetic_app *app)
{
  struct gk_extsts *ext = app->extsts;

  for (int i = 0; i < app->num_species; ++i) {
    if (ext->f_idx[i] < 0) {
      continue;
    }
    struct gk_species *gks = &app->species[i];

    double *f_abs_max = gkyl_calloc_hodev(app->use_gpu, gks->f->ncomp, sizeof(double));
    gkyl_array_reduce_range(f_abs_max, gks->f, GKYL_ABS_MAX, &gks->local);
    double f0_abs_max_local[1];
    if (app->use_gpu) {
      gkyl_cu_memcpy(f0_abs_max_local, f_abs_max, sizeof(double), GKYL_CU_MEMCPY_D2H);
    } else {
      f0_abs_max_local[0] = f_abs_max[0];
    }
    gkyl_free_hodev(app->use_gpu, f_abs_max);

    double f0_abs_max = f0_abs_max_local[0];
    gkyl_comm_allreduce_host(
      app->comm, GKYL_DOUBLE, GKYL_MAX, 1, f0_abs_max_local, &f0_abs_max
    );
    if (!(f0_abs_max > 0.0)) {
      gkyl_exit("ExtSTS: cannot set the absolute tolerance of a species whose f is zero.");
    }
    ext->atol[ext->f_idx[i]] = ext->atol_frac * f0_abs_max;
  }
}

static struct gkyl_update_status
gyrokinetic_update_extsts(gkyl_gyrokinetic_app *app, double dt0)
{
  // Take a time step with the ExtSTS method. The time step is at most dt0, and
  // either the stable time step of the explicit terms (from their latest
  // evaluation) or the one chosen by ARKODE's error control.
  struct gk_extsts *ext = app->extsts;
  struct gkyl_gyrokinetic_fdot_args *args = &app->fdot_args;
  struct gkyl_update_status st = {.success = true};

  double tcurr = app->tcurr;
  double dt = dt0, tout = tcurr + dt0;
  int status = ext->step_begin(app, tcurr, dt0, &dt, &tout);
  if (status < 0) {
    gkyl_gyrokinetic_app_cout(app, stderr, "ExtSTS: step setup failed with status %d.\n", status);
    st.success = false;
    return st;
  }

  for (int i = 0; i < app->num_species; ++i) {
    struct gk_species *gks = &app->species[i];
    args->smom_in[i] = gks->step_moms.f;
    ext->f_state[i] = gks->f;
  }

  for (int i = 0; i < app->num_species; ++i) {
    struct gk_species *gks = &app->species[i];
    gk_species_source_adapt(app, gks, &gks->src, gks->lte.f_lte, args->smom_in, tcurr);
    // Compute moment of f_old to later compute moment of df/dt.
    gk_species_calc_integ_mom_dt(app, gks, dt, gks->fdot_mom_old);
    // Keep the time integral of the stepped moments at the start of the step.
    if (ext->f_idx[i] >= 0) {
      gk_stepped_moms_copy(&gks->step_moms, gks->step_moms.fnew, gks->step_moms.f1);
    }
  }

  // Store the field energies divided by dt for the energy rate and balance diagnostics.
  gk_field_calc_energy_dt(app, app->field, dt, true);

  double tnew = tcurr;
  status = ARKodeEvolve(ext->arkode_mem, tout, ext->state, &tnew, ARK_ONE_STEP);
  if (status < 0) {
    gkyl_gyrokinetic_app_cout(app, stderr, "ExtSTS: time step failed with status %d.\n", status);
    st.success = false;
    return st;
  }
  double dt_actual = tnew - tcurr;

  for (int i = 0; i < app->num_species; ++i) {
    struct gk_species *gks = &app->species[i];
    // Average of the stepped moments over the step.
    if (ext->f_idx[i] >= 0) {
      gk_stepped_moms_combine(
        &gks->step_moms, gks->step_moms.f, 1.0 / dt_actual, gks->step_moms.f1, -1.0 / dt_actual,
        gks->step_moms.fnew
      );
    }
  }

  gyrokinetic_calc_field_and_apply_bc(app, tnew, ext->f_state, args->smom_in, NULL);

  for (int i = 0; i < app->num_species; ++i) {
    struct gk_species *gks = &app->species[i];
    // Compute moment of f_new to compute moment of df/dt.
    // Need to do it after the fields are updated.
    gk_species_calc_integ_mom_dt(app, gks, dt_actual, gks->fdot_mom_new);
  }

  // Store the field energies divided by dt for the energy rate and balance diagnostics.
  gk_field_calc_energy_dt(app, app->field, dt_actual, false);

  // A stage may have a smaller stable time step than the one taken (the step
  // is not retaken). As in the SSP-RK3 stepper, differences below 1% are ignored.
  if (ext->dt_expl_min < 0.99 * dt_actual) {
    ext->num_cfl_exceeded += 1;
  }
  ext->num_steps += 1;

  st.dt_actual = dt_actual;
  st.dt_suggested = ext->step_end(app, dt, dt_actual);
  return st;
}

static struct gkyl_update_status
gyrokinetic_update_extsts_first(gkyl_gyrokinetic_app *app, double dt0)
{
  // First time step: start the integrator from the current time and state (set
  // by the initial conditions or a restart), and get the stable time step.
  struct gk_extsts *ext = app->extsts;

  int status = MRIStepReInitExtSTS(
    ext->arkode_mem, extsts_rhs_diffusive, extsts_rhs_explicit, NULL, app->tcurr, ext->state
  );
  extsts_check(app, status, "MRIStepReInitExtSTS");

  // Evaluate the explicit rhs once, storing it in arrays the SSP-RK3 stepper
  // would use (so that no extra memory is needed).
  int num_arrays = gkyl_nvector_array_num_arrays(ext->state);
  struct gkyl_array *rhs_arrays[num_arrays];
  const struct gkyl_range *rhs_ranges[num_arrays];
  for (int i = 0; i < app->num_species; ++i) {
    struct gk_species *gks = &app->species[i];
    if (ext->f_idx[i] >= 0) {
      rhs_arrays[ext->f_idx[i]] = gks->f1;
      rhs_ranges[ext->f_idx[i]] = &gks->local;
      for (int m = 0; m < gks->step_moms.num; ++m) {
        rhs_arrays[ext->smom_idx[i] + m] = gks->step_moms.fnew[m];
        rhs_ranges[ext->smom_idx[i] + m] = &ext->empty_range;
      }
    }
  }
  N_Vector rhs = gkyl_nvector_array_inew(
    &(struct gkyl_nvector_array_inp){
      .num_arrays = num_arrays,
      .arrays = rhs_arrays,
      .ranges = rhs_ranges,
      .comm = app->comm,
    },
    ext->sunctx
  );
  ext->dt_expl_min = DBL_MAX;
  extsts_rhs_explicit(app->tcurr, ext->state, rhs, app);
  gkyl_nvector_array_release(rhs);

  if (ext->adaptive) {
    // The tolerances are set from the initial (or restart) state, and the
    // first attempted step is the stable time step of the explicit terms.
    extsts_set_atol(app);
    ext->dt_next = fmin(dt0, ext->dt_expl);
    extsts_check(app, ARKodeSetInitStep(ext->arkode_mem, ext->dt_next), "ARKodeSetInitStep");
  }

  app->update_func = gyrokinetic_update_extsts;
  return gyrokinetic_update_extsts(app, dt0);
}

void
gyrokinetic_extsts_init(
  gkyl_gyrokinetic_app *app, const struct gkyl_gk *gk, bool has_implicit_coll
)
{
  if (app->is_multib || app->num_neut_species > 0 || has_implicit_coll) {
    gkyl_exit(
      "The ExtSTS time stepper does not support multiblock apps, neutral species, nor "
      "implicit collisions."
    );
  }
  for (int i = 0; i < app->num_species; ++i) {
    const struct gkyl_gyrokinetic_species *info = &app->species[i].info;
    if (app->species[i].is_fluid || info->positivity.type || info->scaling.type) {
      gkyl_exit(
        "The ExtSTS time stepper does not support fluid species, positivity enforcement, nor "
        "species scaling."
      );
    }
  }

  const struct gkyl_gyrokinetic_time_stepper_inp *ts_inp = &gk->time_stepper;
  if (ts_inp->dom_eig_fac < 0.0 || ts_inp->rtol < 0.0 || ts_inp->atol_frac < 0.0) {
    gkyl_exit("ExtSTS: dom_eig_fac, rtol and atol_frac cannot be negative.");
  }

  struct gk_extsts *ext = gkyl_malloc(sizeof(*ext));
  *ext = (struct gk_extsts){
    .dom_eig_fac = ts_inp->dom_eig_fac == 0.0 ? 2.5 : ts_inp->dom_eig_fac,
    .dt_expl = DBL_MAX,
    .dt_expl_min = DBL_MAX,
    .adaptive = ts_inp->adaptive,
    .rtol = ts_inp->rtol == 0.0 ? 1.0e-4 : ts_inp->rtol,
    .atol_frac = ts_inp->atol_frac == 0.0 ? 1.0e-6 : ts_inp->atol_frac,
    .dt_next = DBL_MAX,
    .step_begin = ts_inp->adaptive ? extsts_step_begin_adaptive : extsts_step_begin_fixed,
    .step_end = ts_inp->adaptive ? extsts_step_end_adaptive : extsts_step_end_fixed,
  };
  app->extsts = ext;

  // The stepped moments evolve with the state but must not enter its norms.
  gkyl_range_init(&ext->empty_range, 1, (int[]){1}, (int[]){0});

  // Index of the arrays of each species in the state vector: the distributions
  // of the evolving species first, followed by their stepped moments.
  int ns = app->num_species;
  ext->f_idx = gkyl_malloc(ns * sizeof(int));
  ext->smom_idx = gkyl_malloc(ns * sizeof(int));
  ext->f_state = gkyl_malloc(ns * sizeof(struct gkyl_array *));
  ext->smom_rate = gkyl_malloc(ns * sizeof(struct gkyl_array **));
  int num_arrays = 0;
  for (int i = 0; i < ns; ++i) {
    ext->f_idx[i] = app->species[i].info.is_static ? -1 : num_arrays++;
  }
  if (num_arrays == 0) {
    gkyl_exit("The ExtSTS time stepper needs at least one species that is not static.");
  }
  for (int i = 0; i < ns; ++i) {
    int num_smom = ext->f_idx[i] < 0 ? 0 : app->species[i].step_moms.num;
    ext->smom_idx[i] = num_arrays;
    ext->smom_rate[i] = gkyl_malloc(GKYL_MAX2(1, num_smom) * sizeof(struct gkyl_array *));
    num_arrays += num_smom;
  }

  struct gkyl_array *arrays[num_arrays];
  const struct gkyl_range *ranges[num_arrays];
  for (int i = 0; i < ns; ++i) {
    struct gk_species *gks = &app->species[i];
    if (ext->f_idx[i] >= 0) {
      arrays[ext->f_idx[i]] = gks->f;
      ranges[ext->f_idx[i]] = &gks->local;
      for (int m = 0; m < gks->step_moms.num; ++m) {
        arrays[ext->smom_idx[i] + m] = gks->step_moms.f1[m];
        ranges[ext->smom_idx[i] + m] = &ext->empty_range;
      }
    }
  }

  // Absolute tolerances: those of the distributions are set in the first step,
  // and those of the stepped moments are irrelevant (they are not in the norms).
  ext->num_arrays = num_arrays;
  ext->atol = gkyl_malloc(num_arrays * sizeof(double));
  for (int k = 0; k < num_arrays; ++k) {
    ext->atol[k] = 1.0;
  }

  // The vector reduces through the app communicator, so SUNDIALS needs none.
  if (SUNContext_Create(SUN_COMM_NULL, &ext->sunctx) != SUN_SUCCESS) {
    gkyl_exit("ExtSTS: could not create the SUNDIALS context.");
  }
  ext->state = gkyl_nvector_array_inew(
    &(struct gkyl_nvector_array_inp){
      .num_arrays = num_arrays,
      .arrays = arrays,
      .ranges = ranges,
      .comm = app->comm,
    },
    ext->sunctx
  );
  if (ext->state == NULL) {
    gkyl_exit("ExtSTS: could not create the state vector.");
  }

  // The integrator is restarted from the actual initial time and state in the first step.
  ext->arkode_mem = MRIStepCreateExtSTS(
    extsts_rhs_diffusive, extsts_rhs_explicit, NULL, app->tcurr, ext->state, ext->sunctx
  );
  if (ext->arkode_mem == NULL) {
    gkyl_exit("ExtSTS: could not create the ARKODE integrator.");
  }
  extsts_check(app, MRIStepGetSTS(ext->arkode_mem, &ext->sts_mem), "MRIStepGetSTS");
  extsts_check(app, ARKodeSetUserData(ext->arkode_mem, app), "ARKodeSetUserData");
  extsts_check(app, ARKodeSetUserData(ext->sts_mem, app), "ARKodeSetUserData");
  // Output is only needed at the end of each step, so no interpolation vectors.
  extsts_check(
    app, ARKodeSetInterpolantType(ext->arkode_mem, ARK_INTERP_NONE), "ARKodeSetInterpolantType"
  );

  // The dominant eigenvalue of the LBO collisions is estimated from their CFL
  // frequency, which is free, so update it in every step.
  extsts_check(app, LSRKStepSetDomEigFn(ext->sts_mem, extsts_dom_eig), "LSRKStepSetDomEigFn");
  extsts_check(app, LSRKStepSetDomEigFrequency(ext->sts_mem, 1), "LSRKStepSetDomEigFrequency");

  extsts_check(app, ARKodeSetPreStepFn(ext->arkode_mem, extsts_pre_step), "ARKodeSetPreStepFn");
  if (ext->adaptive) {
    extsts_check(app, ARKodeWFtolerances(ext->arkode_mem, extsts_ewt), "ARKodeWFtolerances");
    extsts_check(
      app, ARKodeSetStabilityFn(ext->arkode_mem, extsts_stable_dt, app), "ARKodeSetStabilityFn"
    );
    // The stable time step already includes the CFL fraction of the app.
    extsts_check(app, ARKodeSetCFLFraction(ext->arkode_mem, 1.0), "ARKodeSetCFLFraction");
  }

  app->update_func = gyrokinetic_update_extsts_first;
}

void
gyrokinetic_extsts_release(gkyl_gyrokinetic_app *app)
{
  struct gk_extsts *ext = app->extsts;
  if (ext == NULL) {
    return;
  }

  struct gkyl_nvector_array_mem_stats mem = gkyl_nvector_array_get_mem_stats(ext->state);
  gkyl_gyrokinetic_app_cout(
    app, stdout,
    "ExtSTS: %ld steps, %ld explicit and %ld LBO rhs evaluations, %ld steps exceeded the "
    "explicit CFL.\n",
    ext->num_steps, ext->num_rhs_expl, ext->num_rhs_diff, ext->num_cfl_exceeded
  );
  gkyl_gyrokinetic_app_cout(
    app, stdout,
    "ExtSTS: at most %ld state vectors alive (%ld now, %ld never written), %.4g MB each, "
    "%.4g MB at the peak.\n",
    mem.num_peak, mem.num_live, mem.num_unwritten, mem.vec_bytes / 1.0e6, mem.bytes_peak / 1.0e6
  );
  if (ext->adaptive) {
    long num_attempts = 0, num_err_fails = 0, num_exp_limited = 0, num_acc_limited = 0;
    ARKodeGetNumStepAttempts(ext->arkode_mem, &num_attempts);
    ARKodeGetNumErrTestFails(ext->arkode_mem, &num_err_fails);
    ARKodeGetNumExpSteps(ext->arkode_mem, &num_exp_limited);
    ARKodeGetNumAccSteps(ext->arkode_mem, &num_acc_limited);
    gkyl_gyrokinetic_app_cout(
      app, stdout,
      "ExtSTS: %ld step attempts, %ld error test failures; next step limited by the explicit "
      "stability %ld times, by the accuracy %ld times.\n",
      num_attempts, num_err_fails, num_exp_limited, num_acc_limited
    );
  }

  ARKodeFree(&ext->arkode_mem);
  gkyl_nvector_array_release(ext->state);
  SUNContext_Free(&ext->sunctx);

  for (int i = 0; i < app->num_species; ++i) {
    gkyl_free(ext->smom_rate[i]);
  }
  gkyl_free(ext->smom_rate);
  gkyl_free(ext->atol);
  gkyl_free(ext->f_state);
  gkyl_free(ext->smom_idx);
  gkyl_free(ext->f_idx);
  gkyl_free(ext);
  app->extsts = NULL;
}

#else

void
gyrokinetic_extsts_init(
  gkyl_gyrokinetic_app *app, const struct gkyl_gk *gk, bool has_implicit_coll
)
{
  gkyl_exit(
    "The ExtSTS time stepper needs Gkeyll built with SUNDIALS (./configure --use-sundials=yes)."
  );
}

void
gyrokinetic_extsts_release(gkyl_gyrokinetic_app *app)
{
}

#endif
