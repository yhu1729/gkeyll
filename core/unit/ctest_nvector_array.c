#include <acutest.h>

#ifdef GKYL_HAVE_SUNDIALS

#include <gkyl_alloc.h>
#include <gkyl_array.h>
#include <gkyl_array_ops.h>
#include <gkyl_null_comm.h>
#include <gkyl_nvector_array.h>
#include <gkyl_range.h>
#include <gkyl_util.h>

#include <arkode/arkode.h>
#include <arkode/arkode_lsrkstep.h>
#include <arkode/arkode_mristep.h>

#include <float.h>
#include <math.h>

#define NUM_ARRAYS 2

// Vectors, and host copies of their data, used by the tests.
struct nvarr_fixture {
  bool use_gpu; // Whether the vectors live on the device.
  struct gkyl_range ext, local; // Extended and interior ranges of the first array.
  struct gkyl_range range_b; // Range of the second array.
  const struct gkyl_range *ranges[NUM_ARRAYS]; // Reduction range of each array.
  struct gkyl_comm *comm; // Serial communicator.
  SUNContext sunctx; // SUNDIALS context.
  struct gkyl_array *x_ho[NUM_ARRAYS], *y_ho[NUM_ARRAYS]; // Host reference data.
  struct gkyl_array *buffer_ho[NUM_ARRAYS]; // Host buffers to check results.
  N_Vector x, y, out; // Vectors under test.
};

static struct gkyl_array *
mkarr(bool use_gpu, long nc, long size)
{
  return use_gpu ? gkyl_array_cu_dev_new(GKYL_DOUBLE, nc, size) :
                   gkyl_array_new(GKYL_DOUBLE, nc, size);
}

// Copy the host reference data into a vector.
static void
fixture_set(N_Vector vec, struct gkyl_array *data_ho[NUM_ARRAYS])
{
  for (int i = 0; i < NUM_ARRAYS; ++i) {
    gkyl_array_copy(gkyl_nvector_array_get(vec, i), data_ho[i]);
  }
}

static void
fixture_init(struct nvarr_fixture *fix, bool use_gpu)
{
  fix->use_gpu = use_gpu;

  // First array: 3 components on a 6x5 grid, of which the inner 4x3 is reduced.
  gkyl_range_init(&fix->ext, 2, (int[]){0, 0}, (int[]){5, 4});
  gkyl_sub_range_init(&fix->local, &fix->ext, (int[]){1, 1}, (int[]){4, 3});
  // Second array: 2 components on 4 cells, all of them reduced.
  gkyl_range_init(&fix->range_b, 1, (int[]){1}, (int[]){4});
  fix->ranges[0] = &fix->local;
  fix->ranges[1] = &fix->range_b;

  fix->comm = gkyl_null_comm_inew(&(struct gkyl_null_comm_inp){.use_gpu = use_gpu});
  TEST_CHECK(SUNContext_Create(SUN_COMM_NULL, &fix->sunctx) == SUN_SUCCESS);

  long ncomp[NUM_ARRAYS] = {3, 2};
  long size[NUM_ARRAYS] = {fix->ext.volume, fix->range_b.volume};
  struct gkyl_array *x_arrs[NUM_ARRAYS], *y_arrs[NUM_ARRAYS];

  for (int i = 0; i < NUM_ARRAYS; ++i) {
    fix->x_ho[i] = mkarr(false, ncomp[i], size[i]);
    fix->y_ho[i] = mkarr(false, ncomp[i], size[i]);
    fix->buffer_ho[i] = mkarr(false, ncomp[i], size[i]);
    for (long j = 0; j < size[i]; ++j) {
      double *x_d = gkyl_array_fetch(fix->x_ho[i], j);
      double *y_d = gkyl_array_fetch(fix->y_ho[i], j);
      for (long k = 0; k < ncomp[i]; ++k) {
        // Nonzero values of both signs in x, positive values in y.
        x_d[k] = (j % 2 ? -1.0 : 1.0) * (1.0 + 0.01 * j + 0.1 * k + 0.3 * i);
        y_d[k] = 2.0 + 0.02 * j - 0.05 * k + 0.1 * i;
      }
    }
    x_arrs[i] = mkarr(use_gpu, ncomp[i], size[i]);
    y_arrs[i] = mkarr(use_gpu, ncomp[i], size[i]);
  }

  // Large ghost-cell values in the first array must not enter the reductions.
  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, &fix->ext);
  while (gkyl_range_iter_next(&iter)) {
    bool is_ghost = iter.idx[0] < 1 || iter.idx[0] > 4 || iter.idx[1] < 1 || iter.idx[1] > 3;
    if (is_ghost) {
      double *x_d = gkyl_array_fetch(fix->x_ho[0], gkyl_range_idx(&fix->ext, iter.idx));
      for (int k = 0; k < 3; ++k) {
        x_d[k] *= 1.0e10;
      }
    }
  }

  struct gkyl_nvector_array_inp inp = {
    .num_arrays = NUM_ARRAYS,
    .arrays = x_arrs,
    .ranges = fix->ranges,
    .comm = fix->comm,
  };
  fix->x = gkyl_nvector_array_inew(&inp, fix->sunctx);
  inp.arrays = y_arrs;
  fix->y = gkyl_nvector_array_inew(&inp, fix->sunctx);
  TEST_ASSERT(fix->x != NULL && fix->y != NULL);
  fix->out = N_VClone(fix->x);
  TEST_ASSERT(fix->out != NULL);

  // The vectors hold their own references to the arrays.
  for (int i = 0; i < NUM_ARRAYS; ++i) {
    gkyl_array_release(x_arrs[i]);
    gkyl_array_release(y_arrs[i]);
  }

  fixture_set(fix->x, fix->x_ho);
  fixture_set(fix->y, fix->y_ho);
}

static void
fixture_release(struct nvarr_fixture *fix)
{
  gkyl_nvector_array_release(fix->out);
  gkyl_nvector_array_release(fix->y);
  gkyl_nvector_array_release(fix->x);
  for (int i = 0; i < NUM_ARRAYS; ++i) {
    gkyl_array_release(fix->x_ho[i]);
    gkyl_array_release(fix->y_ho[i]);
    gkyl_array_release(fix->buffer_ho[i]);
  }
  SUNContext_Free(&fix->sunctx);
  gkyl_comm_release(fix->comm);
}

// Expected value of an elementwise operation, given the x and y entries.
typedef double (*expected_func_t)(double x, double y, const double *par);

static double
expected_linear_sum(double x, double y, const double *par)
{
  return par[0] * x + par[1] * y;
}
static double
expected_const(double x, double y, const double *par)
{
  return par[0];
}
static double
expected_prod(double x, double y, const double *par)
{
  return x * y;
}
static double
expected_div(double x, double y, const double *par)
{
  return x / y;
}
static double
expected_scale(double x, double y, const double *par)
{
  return par[0] * x;
}
static double
expected_abs(double x, double y, const double *par)
{
  return fabs(x);
}
static double
expected_inv(double x, double y, const double *par)
{
  return 1.0 / x;
}
static double
expected_add_const(double x, double y, const double *par)
{
  return x + par[0];
}

// Check every entry of a vector (ghost cells included) against the expected
// result of an elementwise operation on the reference data.
static void
check_elementwise(
  const struct nvarr_fixture *fix, N_Vector vec, expected_func_t expected, const double *par,
  const char *name
)
{
  for (int i = 0; i < NUM_ARRAYS; ++i) {
    gkyl_array_copy(fix->buffer_ho[i], gkyl_nvector_array_cget(vec, i));
    const double *res_d = fix->buffer_ho[i]->data;
    const double *x_d = fix->x_ho[i]->data;
    const double *y_d = fix->y_ho[i]->data;
    long num_entries = fix->buffer_ho[i]->size * fix->buffer_ho[i]->ncomp;
    for (long j = 0; j < num_entries; ++j) {
      double target = expected(x_d[j], y_d[j], par);
      TEST_CHECK(gkyl_compare(res_d[j], target, 1e-13));
      TEST_MSG("%s array %d entry %ld: %.17g != %.17g", name, i, j, res_d[j], target);
    }
  }
}

static void
nvector_array_elementwise_test(bool use_gpu)
{
  struct nvarr_fixture fix;
  fixture_init(&fix, use_gpu);
  N_Vector x = fix.x, y = fix.y, out = fix.out;

  TEST_CHECK(N_VGetVectorID(x) == SUNDIALS_NVEC_CUSTOM);
  TEST_CHECK(gkyl_nvector_array_num_arrays(x) == NUM_ARRAYS);
  // 12 interior cells with 3 components, plus 4 cells with 2 components.
  TEST_CHECK(N_VGetLength(x) == 44);
  TEST_CHECK(N_VGetLength(out) == 44);

  // Linear sum, with separate output and with every aliasing combination.
  N_VLinearSum(2.0, x, -3.0, y, out);
  check_elementwise(&fix, out, expected_linear_sum, (double[]){2.0, -3.0}, "linear_sum");

  N_VLinearSum(0.5, x, 1.0, y, y);
  check_elementwise(&fix, y, expected_linear_sum, (double[]){0.5, 1.0}, "linear_sum y+=");
  fixture_set(y, fix.y_ho);

  N_VLinearSum(1.0, x, -0.25, y, x);
  check_elementwise(&fix, x, expected_linear_sum, (double[]){1.0, -0.25}, "linear_sum x+=");
  fixture_set(x, fix.x_ho);

  N_VLinearSum(2.0, x, 3.0, y, y);
  check_elementwise(&fix, y, expected_linear_sum, (double[]){2.0, 3.0}, "linear_sum y alias");
  fixture_set(y, fix.y_ho);

  N_VLinearSum(2.0, x, 3.0, y, x);
  check_elementwise(&fix, x, expected_linear_sum, (double[]){2.0, 3.0}, "linear_sum x alias");
  fixture_set(x, fix.x_ho);

  N_VLinearSum(1.0, x, 1.0, x, x);
  check_elementwise(&fix, x, expected_scale, (double[]){2.0}, "linear_sum all alias");
  fixture_set(x, fix.x_ho);

  N_VConst(-1.5, out);
  check_elementwise(&fix, out, expected_const, (double[]){-1.5}, "const");

  N_VProd(x, y, out);
  check_elementwise(&fix, out, expected_prod, NULL, "prod");

  N_VDiv(x, y, out);
  check_elementwise(&fix, out, expected_div, NULL, "div");

  N_VScale(-0.7, x, out);
  check_elementwise(&fix, out, expected_scale, (double[]){-0.7}, "scale");
  N_VScale(3.0, x, x);
  check_elementwise(&fix, x, expected_scale, (double[]){3.0}, "scale in place");
  fixture_set(x, fix.x_ho);

  N_VAbs(x, out);
  check_elementwise(&fix, out, expected_abs, NULL, "abs");

  N_VInv(x, out);
  check_elementwise(&fix, out, expected_inv, NULL, "inv");

  N_VAddConst(x, 0.75, out);
  check_elementwise(&fix, out, expected_add_const, (double[]){0.75}, "add_const");
  N_VAddConst(x, -2.0, x);
  check_elementwise(&fix, x, expected_add_const, (double[]){-2.0}, "add_const in place");
  fixture_set(x, fix.x_ho);

  // The fallback SUNDIALS uses when the vector has no fused operation.
  N_Vector vecs[2] = {x, y};
  TEST_CHECK(N_VLinearCombination(2, (sunrealtype[]){2.0, -3.0}, vecs, out) == SUN_SUCCESS);
  check_elementwise(&fix, out, expected_linear_sum, (double[]){2.0, -3.0}, "linear_combination");

  fixture_release(&fix);
}

static void
nvector_array_reduction_test(bool use_gpu)
{
  struct nvarr_fixture fix;
  fixture_init(&fix, use_gpu);

  // Host reference reductions over the reduction ranges only.
  double dot = 0.0, max_abs = 0.0, min_x = DBL_MAX, wsq_sum = 0.0;
  for (int i = 0; i < NUM_ARRAYS; ++i) {
    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, fix.ranges[i]);
    while (gkyl_range_iter_next(&iter)) {
      long linidx = gkyl_range_idx(fix.ranges[i], iter.idx);
      const double *x_d = gkyl_array_cfetch(fix.x_ho[i], linidx);
      const double *y_d = gkyl_array_cfetch(fix.y_ho[i], linidx);
      for (size_t k = 0; k < fix.x_ho[i]->ncomp; ++k) {
        dot += x_d[k] * y_d[k];
        max_abs = fmax(max_abs, fabs(x_d[k]));
        min_x = fmin(min_x, x_d[k]);
        wsq_sum += (x_d[k] * y_d[k]) * (x_d[k] * y_d[k]);
      }
    }
  }
  double wrms = sqrt(wsq_sum / 44.0);

  // Extrema are exact; sums may differ in the order of the additions.
  TEST_CHECK(gkyl_compare(N_VDotProd(fix.x, fix.y), dot, 1e-12));
  TEST_MSG("dot_prod: %.17g != %.17g", N_VDotProd(fix.x, fix.y), dot);
  TEST_CHECK(N_VMaxNorm(fix.x) == max_abs);
  TEST_MSG("max_norm: %.17g != %.17g", N_VMaxNorm(fix.x), max_abs);
  TEST_CHECK(N_VMin(fix.x) == min_x);
  TEST_MSG("min: %.17g != %.17g", N_VMin(fix.x), min_x);
  TEST_CHECK(gkyl_compare(N_VWrmsNorm(fix.x, fix.y), wrms, 1e-12));
  TEST_MSG("wrms_norm: %.17g != %.17g", N_VWrmsNorm(fix.x, fix.y), wrms);

  // The norm of a constant vector with unit weights is the constant.
  N_VConst(1.0, fix.out);
  N_VConst(-3.0, fix.y);
  TEST_CHECK(gkyl_compare(N_VWrmsNorm(fix.y, fix.out), 3.0, 1e-14));

  fixture_release(&fix);
}

static void
nvector_array_mem_stats_test(bool use_gpu)
{
  struct nvarr_fixture fix;
  fixture_init(&fix, use_gpu);

  // y is the only member of its family; x has one clone (out), never used, so
  // its arrays are not allocated.
  size_t vec_bytes = (30 * 3 + 4 * 2) * sizeof(double);
  struct gkyl_nvector_array_mem_stats stats = gkyl_nvector_array_get_mem_stats(fix.y);
  TEST_CHECK(stats.num_live == 1 && stats.num_peak == 1 && stats.num_created == 1);
  TEST_CHECK(stats.num_unwritten == 0);
  TEST_CHECK(stats.num_allocated == 1 && stats.num_allocated_peak == 1);
  TEST_CHECK(stats.vec_bytes == vec_bytes);
  TEST_CHECK(stats.bytes_live == vec_bytes && stats.bytes_peak == vec_bytes);
  TEST_CHECK(stats.use_gpu == use_gpu);

  stats = gkyl_nvector_array_get_mem_stats(fix.x);
  TEST_CHECK(stats.num_live == 2 && stats.num_peak == 2 && stats.num_created == 2);
  TEST_CHECK(stats.num_unwritten == 1);
  TEST_CHECK(stats.num_allocated == 1 && stats.bytes_live == vec_bytes);

  // Clones (also of clones) join the family; any member reports the same numbers.
  N_Vector clone1 = N_VClone(fix.x);
  N_Vector clone2 = N_VClone(clone1);
  stats = gkyl_nvector_array_get_mem_stats(clone2);
  TEST_CHECK(stats.num_live == 4 && stats.num_peak == 4 && stats.num_created == 4);
  TEST_CHECK(stats.num_unwritten == 3);
  TEST_CHECK(stats.num_allocated == 1 && stats.num_allocated_peak == 1);
  TEST_CHECK(stats.bytes_live == vec_bytes && stats.bytes_peak == vec_bytes);

  // Accessing the data allocates it (zeroed); reading does not count as writing.
  gkyl_nvector_array_cget(clone1, 0);
  TEST_CHECK(N_VMaxNorm(clone1) == 0.0);
  stats = gkyl_nvector_array_get_mem_stats(fix.x);
  TEST_CHECK(stats.num_unwritten == 3 && !gkyl_nvector_array_is_written(clone1));
  TEST_CHECK(stats.num_allocated == 2 && stats.bytes_live == 2 * vec_bytes);
  N_VConst(1.0, clone1);
  N_VConst(2.0, clone1);
  TEST_CHECK(gkyl_nvector_array_is_written(clone1));
  TEST_CHECK(gkyl_nvector_array_is_written(fix.x));
  TEST_CHECK(gkyl_nvector_array_get_mem_stats(fix.x).num_unwritten == 2);
  TEST_CHECK(gkyl_nvector_array_get_mem_stats(fix.x).num_allocated == 2);
  // A clone of an unallocated clone gets the shapes of the family.
  const struct gkyl_array *arr = gkyl_nvector_array_get(clone2, 1);
  TEST_CHECK(arr->ncomp == 2 && arr->size == 4 && gkyl_array_is_cu_dev(arr) == use_gpu);
  stats = gkyl_nvector_array_get_mem_stats(fix.x);
  TEST_CHECK(stats.num_unwritten == 1);
  TEST_CHECK(stats.num_allocated == 3 && stats.num_allocated_peak == 3);

  // Destroying vectors lowers the live counts but not the peaks.
  N_VDestroy(clone1);
  stats = gkyl_nvector_array_get_mem_stats(fix.x);
  TEST_CHECK(stats.num_live == 3 && stats.num_peak == 4 && stats.num_created == 4);
  TEST_CHECK(stats.num_unwritten == 1);
  TEST_CHECK(stats.num_allocated == 2 && stats.num_allocated_peak == 3);
  TEST_CHECK(stats.bytes_live == 2 * vec_bytes && stats.bytes_peak == 3 * vec_bytes);

  gkyl_nvector_array_reset_mem_peak(fix.x);
  stats = gkyl_nvector_array_get_mem_stats(fix.x);
  TEST_CHECK(stats.num_peak == 3 && stats.num_allocated_peak == 2);
  TEST_CHECK(stats.bytes_peak == 2 * vec_bytes);

  // The bookkeeping outlives the vector the family started from.
  N_Vector root = fix.x;
  fix.x = clone2;
  N_VDestroy(root);
  stats = gkyl_nvector_array_get_mem_stats(clone2);
  TEST_CHECK(stats.num_live == 2 && stats.num_created == 4);
  // Only out has never been written, nor allocated.
  TEST_CHECK(stats.num_unwritten == 1 && stats.num_allocated == 1);

  fixture_release(&fix);
}

// Decay rates of the test problem dy/dt = -(RATE_DIFF + RATE_EXPL)*y.
#define RATE_DIFF 20.0
#define RATE_EXPL 1.0

static int
rhs_scale(double fac, N_Vector y, N_Vector rhs)
{
  for (int i = 0; i < gkyl_nvector_array_num_arrays(y); ++i) {
    gkyl_array_set(gkyl_nvector_array_get(rhs, i), fac, gkyl_nvector_array_cget(y, i));
  }
  return 0;
}

static int
rhs_diff(sunrealtype t, N_Vector y, N_Vector rhs, void *user_data)
{
  return rhs_scale(-RATE_DIFF, y, rhs);
}

static int
rhs_expl(sunrealtype t, N_Vector y, N_Vector rhs, void *user_data)
{
  return rhs_scale(-RATE_EXPL, y, rhs);
}

static int
dom_eig(
  sunrealtype t, N_Vector y, N_Vector fn, sunrealtype *lambda_real, sunrealtype *lambda_imag,
  void *user_data, N_Vector temp1, N_Vector temp2, N_Vector temp3
)
{
  *lambda_real = -RATE_DIFF;
  *lambda_imag = 0.0;
  return 0;
}

// Integrate the test problem to t_end with num_steps fixed ExtSTS steps and
// return the maximum error. Also return the memory statistics after the solve.
static double
extsts_solve(
  struct nvarr_fixture *fix, double t_end, int num_steps, struct gkyl_nvector_array_mem_stats *stats
)
{
  N_Vector y = fix->y;
  N_VConst(1.0, y);

  void *arkode_mem = MRIStepCreateExtSTS(rhs_diff, rhs_expl, NULL, 0.0, y, fix->sunctx);
  TEST_ASSERT(arkode_mem != NULL);
  void *sts_mem = NULL;
  TEST_CHECK(MRIStepGetSTS(arkode_mem, &sts_mem) == ARK_SUCCESS);
  TEST_CHECK(LSRKStepSetDomEigFn(sts_mem, dom_eig) == ARK_SUCCESS);
  TEST_CHECK(ARKodeSetFixedStep(arkode_mem, t_end / num_steps) == ARK_SUCCESS);
  TEST_CHECK(ARKodeSetInterpolantType(arkode_mem, ARK_INTERP_NONE) == ARK_SUCCESS);
  TEST_CHECK(ARKodeSetStopTime(arkode_mem, t_end) == ARK_SUCCESS);

  double t_curr = 0.0;
  int status = ARKodeEvolve(arkode_mem, t_end, y, &t_curr, ARK_NORMAL);
  TEST_CHECK(status >= 0);
  TEST_MSG("ARKodeEvolve returned %d", status);
  TEST_CHECK(gkyl_compare(t_curr, t_end, 1e-12));

  *stats = gkyl_nvector_array_get_mem_stats(y);
  ARKodeFree(&arkode_mem);

  // Maximum error with respect to the exact solution.
  N_VAddConst(y, -exp(-(RATE_DIFF + RATE_EXPL) * t_end), fix->out);
  return N_VMaxNorm(fix->out);
}

static void
nvector_array_extsts_test(bool use_gpu)
{
  // Solve a linear decay problem with the ExtSTS method of ARKODE, splitting
  // the rate into a super-time-stepped part and an explicit part.
  struct nvarr_fixture fix;
  fixture_init(&fix, use_gpu);
  // Use a clone of y for the error, so that all work vectors are in y's family.
  N_VDestroy(fix.out);
  fix.out = N_VClone(fix.y);

  double t_end = 0.1;
  int num_steps[3] = {4, 8, 16};
  double error[3];
  struct gkyl_nvector_array_mem_stats stats;
  for (int i = 0; i < 3; ++i) {
    error[i] = extsts_solve(&fix, t_end, num_steps[i], &stats);

    // The integrator cloned work vectors, and released them all when freed.
    TEST_CHECK(stats.num_peak > 2 && stats.num_live <= stats.num_peak);
    TEST_CHECK(stats.num_allocated_peak > 2 && stats.num_allocated_peak <= stats.num_peak);
    TEST_CHECK(stats.bytes_peak == stats.num_allocated_peak * stats.vec_bytes);
    TEST_MSG(
      "%ld vectors alive at the peak, %ld allocated", stats.num_peak, stats.num_allocated_peak
    );
    TEST_CHECK(gkyl_nvector_array_get_mem_stats(fix.y).num_live == 2);
    gkyl_nvector_array_reset_mem_peak(fix.y);
  }

  // The method is second order: halving the step divides the error by ~4.
  for (int i = 0; i < 2; ++i) {
    double order = log2(error[i] / error[i + 1]);
    TEST_CHECK(order > 1.7 && order < 2.3);
    TEST_MSG("errors %g -> %g: observed order %g", error[i], error[i + 1], order);
  }

  fixture_release(&fix);
}

void
test_nvector_array_elementwise_ho(void)
{
  nvector_array_elementwise_test(false);
}

void
test_nvector_array_reduction_ho(void)
{
  nvector_array_reduction_test(false);
}

void
test_nvector_array_mem_stats_ho(void)
{
  nvector_array_mem_stats_test(false);
}

void
test_nvector_array_extsts_ho(void)
{
  nvector_array_extsts_test(false);
}

#ifdef GKYL_HAVE_CUDA
void
test_nvector_array_elementwise_dev(void)
{
  nvector_array_elementwise_test(true);
}

void
test_nvector_array_reduction_dev(void)
{
  nvector_array_reduction_test(true);
}

void
test_nvector_array_mem_stats_dev(void)
{
  nvector_array_mem_stats_test(true);
}

void
test_nvector_array_extsts_dev(void)
{
  nvector_array_extsts_test(true);
}
#endif

#endif

TEST_LIST = {
#ifdef GKYL_HAVE_SUNDIALS
  {"nvector_array_elementwise_ho", test_nvector_array_elementwise_ho},
  {"nvector_array_reduction_ho", test_nvector_array_reduction_ho},
  {"nvector_array_mem_stats_ho", test_nvector_array_mem_stats_ho},
  {"nvector_array_extsts_ho", test_nvector_array_extsts_ho},
#ifdef GKYL_HAVE_CUDA
  {"nvector_array_elementwise_dev", test_nvector_array_elementwise_dev},
  {"nvector_array_reduction_dev", test_nvector_array_reduction_dev},
  {"nvector_array_mem_stats_dev", test_nvector_array_mem_stats_dev},
  {"nvector_array_extsts_dev", test_nvector_array_extsts_dev},
#endif
#endif
  {NULL, NULL}
};
