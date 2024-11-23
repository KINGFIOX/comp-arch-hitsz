#include <omp.h>

#include <cstring>

#include "gemm_kernel_opt.h"
#include "openmp_gemm.h"

void openmp_gemm_opt(int thread_num, float *C, float *A, float *B, uint64_t M, uint64_t N, uint64_t K) {
  // TODO: 练习3的性能优化任务
}
