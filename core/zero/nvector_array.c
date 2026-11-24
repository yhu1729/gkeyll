#ifdef GKYL_HAVE_SUNDIALS

#include <gkyl_alloc.h>
#include <gkyl_array_ops.h>
#include <gkyl_array_reduce.h>
#include <gkyl_nvector_array.h>
#include <gkyl_util.h>

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

_Static_assert(sizeof(sunrealtype) == sizeof(double), "SUNDIALS must use double precision");

// Memory bookkeeping, and shape of the arrays, shared by a vector and all the
// vectors cloned from it.
struct nvarr_mem_monitor {
  long num_live; // Vectors alive now (also the reference count of this struct).
  long num_peak; // Largest number of vectors alive at the same time.
  long num_created; // Vectors created so far.
  long num_unwritten; // Live vectors that have never been written to.
  long num_allocated; // Live vectors whose arrays are allocated.
  long num_allocated_peak; // Largest number of vectors allocated at the same time.
  size_t vec_bytes; // Bytes of array data in one vector.
  size_t bytes_peak; // Largest number of bytes allocated at the same time.
  int num_arrays; // Number of arrays in each vector.
  size_t *ncomp; // Number of components of each array.
  size_t *size; // Number of cells of each array.
};

// A clone allocates its arrays when they are first accessed, so a work vector
// that a SUNDIALS integrator creates but never uses takes no array memory.
struct nvarr_content {
  int num_arrays; // Number of arrays.
  struct gkyl_array **arrays; // Arrays holding the data (owned references; NULL until allocated).
  struct gkyl_range *ranges; // Range each array is reduced over.
  struct gkyl_comm *comm; // Communicator used to reduce across ranks.
  bool use_gpu; // Whether the arrays live on the device.
  int64_t global_length; // Number of entries reduced over, across all ranks.
  size_t max_ncomp; // Largest number of components among the arrays.
  double *red_buffer; // Per-component reductions, on the backend of the arrays.
  double *red_buffer_ho; // Per-component reductions on the host.
  bool is_allocated; // Whether the arrays are allocated.
  bool is_written; // Whether the data has been written to.
  struct nvarr_mem_monitor *monitor; // Memory bookkeeping of the family of clones.
};

static void
nvarr_monitor_count_allocated(struct nvarr_mem_monitor *monitor)
{
  monitor->num_allocated += 1;
  monitor->num_allocated_peak = GKYL_MAX2(monitor->num_allocated_peak, monitor->num_allocated);
  monitor->bytes_peak =
    GKYL_MAX2(monitor->bytes_peak, monitor->num_allocated * monitor->vec_bytes);
}

// Allocate the arrays of a clone, zeroed: elementwise operations also act on
// ghost cells, so every cell must hold a valid number.
static void
nvarr_alloc_arrays(struct nvarr_content *content)
{
  const struct nvarr_mem_monitor *monitor = content->monitor;
  for (int i = 0; i < content->num_arrays; ++i) {
    content->arrays[i] =
      content->use_gpu ? gkyl_array_cu_dev_new(GKYL_DOUBLE, monitor->ncomp[i], monitor->size[i]) :
                         gkyl_array_new(GKYL_DOUBLE, monitor->ncomp[i], monitor->size[i]);
    gkyl_array_clear(content->arrays[i], 0.0);
  }
  content->is_allocated = true;
  nvarr_monitor_count_allocated(content->monitor);
}

static inline struct nvarr_content *
nvarr_get_content(N_Vector vec)
{
  return vec->content;
}

// Content of a vector whose arrays are about to be read or written: allocates
// the arrays on the first access.
static inline struct nvarr_content *
nvarr_get_content_data(N_Vector vec)
{
  struct nvarr_content *content = vec->content;
  if (!content->is_allocated) {
    nvarr_alloc_arrays(content);
  }
  return content;
}

static inline void
nvarr_mark_written(struct nvarr_content *content)
{
  if (!content->is_written) {
    content->is_written = true;
    content->monitor->num_unwritten -= 1;
  }
}

static void
nvarr_monitor_add(struct nvarr_mem_monitor *monitor, bool is_written)
{
  monitor->num_live += 1;
  monitor->num_created += 1;
  monitor->num_unwritten += is_written ? 0 : 1;
  monitor->num_peak = GKYL_MAX2(monitor->num_peak, monitor->num_live);
}

static void
nvarr_monitor_release(struct nvarr_mem_monitor *monitor)
{
  gkyl_free(monitor->size);
  gkyl_free(monitor->ncomp);
  gkyl_free(monitor);
}

// Allocate the parts of the content that do not depend on who owns the arrays.
static struct nvarr_content *
nvarr_content_new(int num_arrays, const struct gkyl_comm *comm, bool use_gpu, size_t max_ncomp)
{
  struct nvarr_content *content = gkyl_malloc(sizeof(*content));

  content->num_arrays = num_arrays;
  content->arrays = gkyl_malloc(num_arrays * sizeof(struct gkyl_array *));
  content->ranges = gkyl_malloc(num_arrays * sizeof(struct gkyl_range));
  content->comm = gkyl_comm_acquire(comm);
  content->use_gpu = use_gpu;
  content->max_ncomp = max_ncomp;
  content->red_buffer_ho = gkyl_malloc(max_ncomp * sizeof(double));
  content->red_buffer = use_gpu ? gkyl_cu_malloc(max_ncomp * sizeof(double)) :
                                  content->red_buffer_ho;

  return content;
}

static N_Vector_ID
nvarr_get_vector_id(N_Vector vec)
{
  return SUNDIALS_NVEC_CUSTOM;
}

static N_Vector
nvarr_clone(N_Vector vec)
{
  const struct nvarr_content *content = nvarr_get_content(vec);

  N_Vector clone = N_VNewEmpty(vec->sunctx);
  if (clone == NULL) {
    return NULL;
  }
  if (N_VCopyOps(vec, clone) != SUN_SUCCESS) {
    N_VFreeEmpty(clone);
    return NULL;
  }

  // The arrays are allocated on first access (see nvarr_get_content_data).
  struct nvarr_content *clone_content =
    nvarr_content_new(content->num_arrays, content->comm, content->use_gpu, content->max_ncomp);
  for (int i = 0; i < content->num_arrays; ++i) {
    clone_content->arrays[i] = NULL;
    clone_content->ranges[i] = content->ranges[i];
  }
  clone_content->global_length = content->global_length;
  clone_content->is_allocated = false;
  clone_content->is_written = false;
  clone_content->monitor = content->monitor;
  nvarr_monitor_add(clone_content->monitor, false);

  clone->content = clone_content;
  return clone;
}

static void
nvarr_destroy(N_Vector vec)
{
  if (vec == NULL) {
    return;
  }

  struct nvarr_content *content = nvarr_get_content(vec);
  if (content != NULL) {
    struct nvarr_mem_monitor *monitor = content->monitor;
    monitor->num_live -= 1;
    monitor->num_unwritten -= content->is_written ? 0 : 1;
    monitor->num_allocated -= content->is_allocated ? 1 : 0;
    if (monitor->num_live == 0) {
      nvarr_monitor_release(monitor);
    }

    for (int i = 0; i < content->num_arrays; ++i) {
      gkyl_array_release(content->arrays[i]); // NULL if never allocated.
    }
    if (content->use_gpu) {
      gkyl_cu_free(content->red_buffer);
    }
    gkyl_free(content->red_buffer_ho);
    gkyl_comm_release(content->comm);
    gkyl_free(content->ranges);
    gkyl_free(content->arrays);
    gkyl_free(content);
    vec->content = NULL;
  }

  N_VFreeEmpty(vec);
}

static sunindextype
nvarr_get_length(N_Vector vec)
{
  return nvarr_get_content(vec)->global_length;
}

// out = coeff_x*x + coeff_y*y. out may be x and/or y.
static void
nvarr_linear_sum(sunrealtype coeff_x, N_Vector x, sunrealtype coeff_y, N_Vector y, N_Vector out)
{
  const struct nvarr_content *x_content = nvarr_get_content_data(x);
  const struct nvarr_content *y_content = nvarr_get_content_data(y);
  struct nvarr_content *out_content = nvarr_get_content_data(out);

  for (int i = 0; i < out_content->num_arrays; ++i) {
    const struct gkyl_array *x_arr = x_content->arrays[i];
    const struct gkyl_array *y_arr = y_content->arrays[i];
    struct gkyl_array *out_arr = out_content->arrays[i];

    if (out_arr == y_arr && out_arr != x_arr && coeff_y == 1.0) {
      gkyl_array_accumulate(out_arr, coeff_x, x_arr);
    } else if (out_arr == x_arr && out_arr != y_arr && coeff_x == 1.0) {
      gkyl_array_accumulate(out_arr, coeff_y, y_arr);
    } else {
      gkyl_array_comp_op(out_arr, GKYL_AXPBY, coeff_x, x_arr, coeff_y, y_arr);
    }
  }
  nvarr_mark_written(out_content);
}

static void
nvarr_const(sunrealtype val, N_Vector out)
{
  struct nvarr_content *out_content = nvarr_get_content_data(out);

  for (int i = 0; i < out_content->num_arrays; ++i) {
    gkyl_array_clear(out_content->arrays[i], val);
  }
  nvarr_mark_written(out_content);
}

// out = op(x, y) for a componentwise operator (y is NULL for one-input operators).
static void
nvarr_comp_op(enum gkyl_array_op op, N_Vector x, N_Vector y, N_Vector out)
{
  const struct nvarr_content *x_content = nvarr_get_content_data(x);
  const struct nvarr_content *y_content = y ? nvarr_get_content_data(y) : NULL;
  struct nvarr_content *out_content = nvarr_get_content_data(out);

  for (int i = 0; i < out_content->num_arrays; ++i) {
    gkyl_array_comp_op(
      out_content->arrays[i], op, 1.0, x_content->arrays[i], 0.0,
      y_content ? y_content->arrays[i] : NULL
    );
  }
  nvarr_mark_written(out_content);
}

static void
nvarr_prod(N_Vector x, N_Vector y, N_Vector out)
{
  nvarr_comp_op(GKYL_PROD, x, y, out);
}

static void
nvarr_div(N_Vector x, N_Vector y, N_Vector out)
{
  nvarr_comp_op(GKYL_DIV, x, y, out);
}

static void
nvarr_abs(N_Vector x, N_Vector out)
{
  nvarr_comp_op(GKYL_ABS, x, NULL, out);
}

static void
nvarr_inv(N_Vector x, N_Vector out)
{
  nvarr_comp_op(GKYL_INV, x, NULL, out);
}

static void
nvarr_scale(sunrealtype fac, N_Vector x, N_Vector out)
{
  const struct nvarr_content *x_content = nvarr_get_content_data(x);
  struct nvarr_content *out_content = nvarr_get_content_data(out);

  for (int i = 0; i < out_content->num_arrays; ++i) {
    if (out_content->arrays[i] == x_content->arrays[i]) {
      gkyl_array_scale(out_content->arrays[i], fac);
    } else {
      gkyl_array_set(out_content->arrays[i], fac, x_content->arrays[i]);
    }
  }
  nvarr_mark_written(out_content);
}

static void
nvarr_add_const(N_Vector x, sunrealtype val, N_Vector out)
{
  const struct nvarr_content *x_content = nvarr_get_content_data(x);
  struct nvarr_content *out_content = nvarr_get_content_data(out);

  for (int i = 0; i < out_content->num_arrays; ++i) {
    struct gkyl_array *out_arr = out_content->arrays[i];
    if (out_arr != x_content->arrays[i]) {
      gkyl_array_set(out_arr, 1.0, x_content->arrays[i]);
    }
    // One pass per component: only used to build error weights (once per step).
    for (size_t k = 0; k < out_arr->ncomp; ++k) {
      gkyl_array_shiftc(out_arr, val, k);
    }
  }
  nvarr_mark_written(out_content);
}

// Reduce op(x) (or the weighted op(x, wgt) if wgt is not NULL) over the range
// of every array and over all ranks, combining with comb_op (SUM, MAX or MIN).
static double
nvarr_reduce(
  const struct nvarr_content *x_content, const struct nvarr_content *wgt_content,
  enum gkyl_array_op op, enum gkyl_array_op comb_op
)
{
  double red_local = comb_op == GKYL_SUM ? 0.0 : (comb_op == GKYL_MAX ? -DBL_MAX : DBL_MAX);

  for (int i = 0; i < x_content->num_arrays; ++i) {
    const struct gkyl_array *x_arr = x_content->arrays[i];
    size_t ncomp = x_arr->ncomp;

    if (x_content->ranges[i].volume == 0) {
      continue; // Arrays excluded from the reductions.
    }

    if (wgt_content) {
      gkyl_array_reduce_weighted_range(
        x_content->red_buffer, x_arr, wgt_content->arrays[i], op, &x_content->ranges[i]
      );
    } else {
      gkyl_array_reduce_range(x_content->red_buffer, x_arr, op, &x_content->ranges[i]);
    }
    if (x_content->use_gpu) {
      gkyl_cu_memcpy(
        x_content->red_buffer_ho, x_content->red_buffer, ncomp * sizeof(double), GKYL_CU_MEMCPY_D2H
      );
    }

    for (size_t k = 0; k < ncomp; ++k) {
      double red_comp = x_content->red_buffer_ho[k];
      switch (comb_op) {
        case GKYL_SUM:
          red_local += red_comp;
          break;
        case GKYL_MAX:
          red_local = fmax(red_local, red_comp);
          break;
        case GKYL_MIN:
          red_local = fmin(red_local, red_comp);
          break;
        default:
          assert(false);
          break;
      }
    }
  }

  double red_global = red_local;
  gkyl_comm_allreduce_host(x_content->comm, GKYL_DOUBLE, comb_op, 1, &red_local, &red_global);
  return red_global;
}

static sunrealtype
nvarr_dot_prod(N_Vector x, N_Vector y)
{
  return nvarr_reduce(nvarr_get_content_data(x), nvarr_get_content_data(y), GKYL_SUM, GKYL_SUM);
}

static sunrealtype
nvarr_max_norm(N_Vector x)
{
  return nvarr_reduce(nvarr_get_content_data(x), NULL, GKYL_ABS_MAX, GKYL_MAX);
}

static sunrealtype
nvarr_min(N_Vector x)
{
  return nvarr_reduce(nvarr_get_content_data(x), NULL, GKYL_MIN, GKYL_MIN);
}

// sqrt(sum_i (x_i*wgt_i)^2 / N), with N the global length.
static sunrealtype
nvarr_wrms_norm(N_Vector x, N_Vector wgt)
{
  const struct nvarr_content *x_content = nvarr_get_content_data(x);
  double sq_sum =
    nvarr_reduce(x_content, nvarr_get_content_data(wgt), GKYL_PROD_SQ_SUM, GKYL_SUM);
  return sqrt(sq_sum / x_content->global_length);
}

N_Vector
gkyl_nvector_array_inew(const struct gkyl_nvector_array_inp *inp, SUNContext sunctx)
{
  assert(inp->num_arrays > 0);

  N_Vector vec = N_VNewEmpty(sunctx);
  if (vec == NULL) {
    return NULL;
  }

  vec->ops->nvgetvectorid = nvarr_get_vector_id;
  vec->ops->nvclone = nvarr_clone;
  vec->ops->nvdestroy = nvarr_destroy;
  vec->ops->nvgetlength = nvarr_get_length;
  vec->ops->nvlinearsum = nvarr_linear_sum;
  vec->ops->nvconst = nvarr_const;
  vec->ops->nvprod = nvarr_prod;
  vec->ops->nvdiv = nvarr_div;
  vec->ops->nvscale = nvarr_scale;
  vec->ops->nvabs = nvarr_abs;
  vec->ops->nvinv = nvarr_inv;
  vec->ops->nvaddconst = nvarr_add_const;
  vec->ops->nvdotprod = nvarr_dot_prod;
  vec->ops->nvmaxnorm = nvarr_max_norm;
  vec->ops->nvmin = nvarr_min;
  vec->ops->nvwrmsnorm = nvarr_wrms_norm;

  bool use_gpu = gkyl_array_is_cu_dev(inp->arrays[0]);
  size_t max_ncomp = 0;
  for (int i = 0; i < inp->num_arrays; ++i) {
    assert(inp->arrays[i]->type == GKYL_DOUBLE);
    assert(gkyl_array_is_cu_dev(inp->arrays[i]) == use_gpu);
    max_ncomp = GKYL_MAX2(max_ncomp, inp->arrays[i]->ncomp);
  }

  struct nvarr_content *content = nvarr_content_new(inp->num_arrays, inp->comm, use_gpu, max_ncomp);

  struct nvarr_mem_monitor *monitor = gkyl_malloc(sizeof(*monitor));
  *monitor = (struct nvarr_mem_monitor){
    .num_arrays = inp->num_arrays,
    .ncomp = gkyl_malloc(inp->num_arrays * sizeof(size_t)),
    .size = gkyl_malloc(inp->num_arrays * sizeof(size_t)),
  };

  int64_t local_length = 0;
  for (int i = 0; i < inp->num_arrays; ++i) {
    content->arrays[i] = gkyl_array_acquire(inp->arrays[i]);
    content->ranges[i] = *inp->ranges[i];
    local_length += inp->ranges[i]->volume * (int64_t)inp->arrays[i]->ncomp;
    monitor->ncomp[i] = inp->arrays[i]->ncomp;
    monitor->size[i] = inp->arrays[i]->size;
    monitor->vec_bytes += inp->arrays[i]->size * inp->arrays[i]->esznc;
  }
  content->global_length = local_length;
  gkyl_comm_allreduce_host(
    content->comm, GKYL_INT_64, GKYL_SUM, 1, &local_length, &content->global_length
  );

  content->monitor = monitor;
  // The wrapped arrays hold the caller's data, so they count as allocated and written.
  content->is_allocated = true;
  content->is_written = true;
  nvarr_monitor_add(monitor, true);
  nvarr_monitor_count_allocated(monitor);

  vec->content = content;
  return vec;
}

struct gkyl_array *
gkyl_nvector_array_get(N_Vector vec, int idx)
{
  struct nvarr_content *content = nvarr_get_content_data(vec);
  assert(idx >= 0 && idx < content->num_arrays);
  nvarr_mark_written(content);
  return content->arrays[idx];
}

const struct gkyl_array *
gkyl_nvector_array_cget(N_Vector vec, int idx)
{
  const struct nvarr_content *content = nvarr_get_content_data(vec);
  assert(idx >= 0 && idx < content->num_arrays);
  return content->arrays[idx];
}

int
gkyl_nvector_array_num_arrays(N_Vector vec)
{
  return nvarr_get_content(vec)->num_arrays;
}

struct gkyl_nvector_array_mem_stats
gkyl_nvector_array_get_mem_stats(N_Vector vec)
{
  const struct nvarr_content *content = nvarr_get_content(vec);
  const struct nvarr_mem_monitor *monitor = content->monitor;

  return (struct gkyl_nvector_array_mem_stats){
    .num_live = monitor->num_live,
    .num_peak = monitor->num_peak,
    .num_created = monitor->num_created,
    .num_unwritten = monitor->num_unwritten,
    .num_allocated = monitor->num_allocated,
    .num_allocated_peak = monitor->num_allocated_peak,
    .vec_bytes = monitor->vec_bytes,
    .bytes_live = monitor->num_allocated * monitor->vec_bytes,
    .bytes_peak = monitor->bytes_peak,
    .use_gpu = content->use_gpu,
  };
}

bool
gkyl_nvector_array_is_written(N_Vector vec)
{
  return nvarr_get_content(vec)->is_written;
}

void
gkyl_nvector_array_reset_mem_peak(N_Vector vec)
{
  struct nvarr_mem_monitor *monitor = nvarr_get_content(vec)->monitor;
  monitor->num_peak = monitor->num_live;
  monitor->num_allocated_peak = monitor->num_allocated;
  monitor->bytes_peak = monitor->num_allocated * monitor->vec_bytes;
}

void
gkyl_nvector_array_release(N_Vector vec)
{
  N_VDestroy(vec);
}

#endif
