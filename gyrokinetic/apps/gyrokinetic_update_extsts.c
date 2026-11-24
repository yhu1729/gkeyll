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

  int *f_idx; // Index in the state of the distribution of each species (-1 if static).
  int *smom_idx; // Index in the state of the first stepped moment of each species.
  struct gkyl_range empty_range; // Excludes the stepped moments from norms.
  struct gkyl_array **f_state; // Distribution of each species in the vector being evaluated.
  struct gkyl_array ***smom_rate; // Stepped-moment rates of each species in the rhs vector.

  double dom_eig_fac; // Dominant eigenvalue / LBO CFL frequency.
  double dt_expl; // Stable time step of the latest explicit rhs.
  double dt_expl_min; // Smallest stable time step of the explicit rhs in this step.
  double omega_diff; // LBO CFL frequency of the latest diffusive rhs on this rank.

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

static void
extsts_check(const gkyl_gyrokinetic_app *app, int status, const char *call)
{
  if (status < 0) {
    gkyl_gyrokinetic_app_cout(app, stderr, "ExtSTS: %s failed with status %d.\n", call, status);
    gkyl_exit("ExtSTS time stepper setup failed.");
  }
}

static struct gkyl_update_status
gyrokinetic_update_extsts(gkyl_gyrokinetic_app *app, double dt0)
{
  // Take a time step with the ExtSTS method. The time step is the stable time
  // step of the explicit terms, obtained from their latest evaluation.
  struct gk_extsts *ext = app->extsts;
  struct gkyl_gyrokinetic_fdot_args *args = &app->fdot_args;
  struct gkyl_update_status st = {.success = true};

  double tcurr = app->tcurr;
  double dt = fmin(dt0, ext->dt_expl);

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

  ext->dt_expl_min = DBL_MAX;
  double tnew = tcurr;
  int status = ARKodeSetFixedStep(ext->arkode_mem, dt);
  if (status == ARK_SUCCESS) {
    status = ARKodeEvolve(ext->arkode_mem, tcurr + dt, ext->state, &tnew, ARK_ONE_STEP);
  }
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
  st.dt_suggested = ext->dt_expl;
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

  struct gk_extsts *ext = gkyl_malloc(sizeof(*ext));
  *ext = (struct gk_extsts){
    .dom_eig_fac = gk->time_stepper.dom_eig_fac == 0.0 ? 2.5 : gk->time_stepper.dom_eig_fac,
    .dt_expl = DBL_MAX,
    .dt_expl_min = DBL_MAX,
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

  ARKodeFree(&ext->arkode_mem);
  gkyl_nvector_array_release(ext->state);
  SUNContext_Free(&ext->sunctx);

  for (int i = 0; i < app->num_species; ++i) {
    gkyl_free(ext->smom_rate[i]);
  }
  gkyl_free(ext->smom_rate);
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
