#pragma once

#ifdef GKYL_HAVE_SUNDIALS

#include <gkyl_array.h>
#include <gkyl_comm.h>
#include <gkyl_range.h>

#include <sundials/sundials_nvector.h>

#include <stdbool.h>
#include <stddef.h>

/**
 * SUNDIALS N_Vector whose data is a list of gkyl_arrays (e.g. the distribution
 * function of every species). The vector does not copy the data: its operations
 * act directly on the arrays, on the host or on the device.
 *
 * Elementwise operations act on all allocated cells (including ghost cells),
 * as the Gkeyll time steppers do. Reductions (norms, dot product, min) only
 * include the cells in the range given for each array, and are reduced across
 * the communicator, so ghost cells never enter a norm.
 */

// Inputs to create a new array N_Vector.
struct gkyl_nvector_array_inp {
  int num_arrays; // Number of arrays in the vector.
  struct gkyl_array **arrays; // Double arrays to wrap, all on the same backend.
  const struct gkyl_range **ranges; // Range each array is reduced over.
  struct gkyl_comm *comm; // Communicator used to reduce across ranks.
};

// Memory used by a vector and all the vectors cloned (directly or indirectly)
// from it. Counts are local to this MPI rank.
struct gkyl_nvector_array_mem_stats {
  long num_live; // Vectors alive now, including the one made with _inew.
  long num_peak; // Largest number of vectors alive at the same time.
  long num_created; // Vectors created so far (the _inew vector and every clone).
  long num_unwritten; // Live vectors that have never been written to.
  size_t vec_bytes; // Bytes of array data in one vector.
  size_t bytes_live; // Bytes of array data in the live vectors.
  size_t bytes_peak; // Largest bytes_live reached.
  bool use_gpu; // Whether the array data lives on the device.
};

/**
 * Create a new N_Vector that wraps the input arrays. The vector acquires the
 * arrays and the communicator, and copies the ranges. Vectors cloned from it
 * (e.g. by a SUNDIALS integrator) allocate their own arrays on the same backend.
 *
 * @param inp Input parameters defined in gkyl_nvector_array_inp struct.
 * @param sunctx SUNDIALS context the vector belongs to.
 * @return New vector, or NULL if SUNDIALS could not create it.
 */
N_Vector gkyl_nvector_array_inew(const struct gkyl_nvector_array_inp *inp, SUNContext sunctx);

/**
 * Get an array of the vector for writing (e.g. to store a right-hand side).
 * The vector keeps ownership. This marks the vector as written.
 *
 * @param vec Array N_Vector.
 * @param idx Index of the array in the vector.
 * @return Array idx of the vector.
 */
struct gkyl_array *gkyl_nvector_array_get(N_Vector vec, int idx);

/**
 * Get an array of the vector for reading. The vector keeps ownership.
 *
 * @param vec Array N_Vector.
 * @param idx Index of the array in the vector.
 * @return Array idx of the vector.
 */
const struct gkyl_array *gkyl_nvector_array_cget(N_Vector vec, int idx);

/**
 * Get the number of arrays in the vector.
 *
 * @param vec Array N_Vector.
 * @return Number of arrays.
 */
int gkyl_nvector_array_num_arrays(N_Vector vec);

/**
 * Get the memory used by a vector and by every vector cloned from it, which
 * includes the work vectors a SUNDIALS integrator clones internally.
 *
 * @param vec Array N_Vector (any member of the family of clones).
 * @return Memory statistics of the family of clones on this MPI rank.
 */
struct gkyl_nvector_array_mem_stats gkyl_nvector_array_get_mem_stats(N_Vector vec);

/**
 * Check whether a vector has ever been written to, by a vector operation or
 * through gkyl_nvector_array_get. A clone that is never written is memory a
 * SUNDIALS integrator allocated without using.
 *
 * @param vec Array N_Vector.
 * @return True if the vector has been written to.
 */
bool gkyl_nvector_array_is_written(N_Vector vec);

/**
 * Restart the peak counters of the family of clones from the live values, e.g.
 * to measure the peak of the time loop separately from initialization.
 *
 * @param vec Array N_Vector (any member of the family of clones).
 */
void gkyl_nvector_array_reset_mem_peak(N_Vector vec);

/**
 * Release a vector (equivalent to N_VDestroy).
 *
 * @param vec Array N_Vector to release.
 */
void gkyl_nvector_array_release(N_Vector vec);

#endif
