# arch-hitsz

## lab4

### 实现流程

```c
__global__ void MatrixMulKernel(float *d_M, float *d_N, float *d_P, int width) {
  // Calculate the row index of the P element and M
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  // Calculate the column index of the P element and N
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Ensure the thread is within bounds
  if (row < width && col < width) {
    float pValue = 0.0;

    // Each thread computes one element of the matrix
    for (int k = 0; k < width; ++k) {
      pValue += d_M[row * width + k] * d_N[k * width + col];
    }

    // Store the computed value into the output matrix
    d_P[row * width + col] = pValue;
  }
}

int main(int argc, char * argv[]) {
  // ...
  for (int j = 0; j < nIter; j++) {
    MatrixMulKernel<<<grid, block>>>(d_M, d_N, d_P, m);
  }
  // ...
}
```

### 测试结果

- 编译运行

```sh
# command
lenovo@10:~/comp-arch-hitsz$ ./a.out 1 1000

# output
Kernel Elpased Time: 48.195 ms
Performance= 41.50 GFlop/s, Time= 48.195 msec, Size= 2000000000 Ops
Computing result using host CPU...done.
Listing first 100 Differences > 0.000010...
```

- 只更改矩阵尺寸

```sh
# command
lenovo@10:~/comp-arch-hitsz$ ./a.out 0 3000
# output
Kernel Elpased Time: 1284.377 ms
Performance= 42.04 GFlop/s, Time= 1284.377 msec, Size= 54000000000 Ops

# command
lenovo@10:~/comp-arch-hitsz$ ./a.out 0 5000
# output
Kernel Elpased Time: 5899.872 ms
Performance= 42.37 GFlop/s, Time= 5899.872 msec, Size= 250000000000 Ops

# command
lenovo@10:~/comp-arch-hitsz$ ./a.out 0 10000
# output
Kernel Elpased Time: 44527.219 ms
Performance= 44.92 GFlop/s, Time= 44527.219 msec, Size= 2000000000000 Ops

# command
lenovo@10:~/comp-arch-hitsz$ ./a.out 0 20000
# output
Kernel Elpased Time: -0.000 ms
Performance= -5229295207189833728.00 GFlop/s, Time= -0.000 msec, Size= 16000000000000 Ops
```

发现 size 太大, 会出现计算出错

- 只更改 TILE_SIZE

为了方便, 我改了一下 bash 脚本

```sh
function compile_with_tile_size() {
    local tile_size=$1
    nvcc -arch=compute_35 -L/usr/local/cuda/lib64 -lcublas ./matrix_mul.cu -DTILE_SIZE=$tile_size -o $tile_size
    ./$tile_size 0 1000
}

for tile_size in 64 ; do
    echo "tile_size: $tile_size"
    compile_with_tile_size $tile_size
done
```

下面是跑出来的结果

```sh
# command
lenovo@10:~/comp-arch-hitsz$ ./bench4.sh 

# output
tile_size: 2
Kernel Elpased Time: 4754.647 ms
Performance= 0.42 GFlop/s, Time= 4754.647 msec, Size= 2000000000 Ops
tile_size: 4
Kernel Elpased Time: 959.101 ms
Performance= 2.09 GFlop/s, Time= 959.101 msec, Size= 2000000000 Ops
tile_size: 8
Kernel Elpased Time: 284.650 ms
Performance= 7.03 GFlop/s, Time= 284.650 msec, Size= 2000000000 Ops
tile_size: 16
Kernel Elpased Time: 181.891 ms
Performance= 11.00 GFlop/s, Time= 181.891 msec, Size= 2000000000 Ops
tile_size: 32
Kernel Elpased Time: 154.600 ms
Performance= 12.94 GFlop/s, Time= 154.600 msec, Size= 2000000000 Ops
```

tile_size=64 时, 出现问题

```sh
# command
lenovo@10:~/comp-arch-hitsz$ ./bench4.sh 

# output
tile_size: 64
Kernel Elpased Time: 0.000 ms
Performance= 4807692.10 GFlop/s, Time= 0.000 msec, Size= 2000000000 Ops
```

### 结论

随着矩阵尺寸变大, performance 降低, 随着 tile_size 增大, performace 升高,
但是大于一定值后, 会有所下降

# 分析

1. 随着矩阵尺寸增大，性能降低的原因

- 计算量急剧增加：矩阵乘法的时间复杂度为 \( O(N^3) \)。当矩阵尺寸从 \( N \) 增加到 \( N+1 \) 时，
  计算量并不是线性增加，而是以立方倍数增长。因此，随着矩阵尺寸的增大，计算所需的总操作数迅速增加，导致性能下降。
- 内存带宽和缓存限制：较大的矩阵数据可能无法全部存放在 GPU 的缓存或共享内存中，需要频繁地访问全局内存。
  而全局内存的访问延迟较高，带宽也有限，这会成为性能瓶颈。
- 线程调度和资源竞争：当矩阵尺寸增大时，线程数量和所需的计算资源也增加。
  GPU 的计算核心和内存带宽是有限的，过多的线程可能导致资源争夺，增加线程调度开销，降低计算效率。
- 硬件限制：GPU 的寄存器数量、共享内存大小等硬件资源是有限的。大型矩阵运算可能会超出这些资源的限制，导致性能无法提升，甚至程序无法正确运行。

2. 随着 `tile_size` 增大，性能先升高后下降的原因

- 数据复用提升：增大 `tile_size`（即分块大小）可以使每个线程块加载更多的数据到共享内存中，增加数据的复用率，减少对全局内存的访问次数，从而提高性能。
- 并行度降低：然而，`tile_size` 增大后，每个线程块处理的数据增多，所需的共享内存和寄存器数量也增加。
  这可能导致每个流多处理器（SM）上能同时驻留的线程块数量减少，降低 GPU 的硬件占用率（Occupancy），从而影响性能。
- 共享内存容量限制：GPU 的每个 SM 共享内存容量是固定的。
  当 `tile_size` 增大到一定程度后，单个线程块所需的共享内存可能超过硬件限制，导致无法分配足够的共享内存，程序可能出现错误或性能下降。
- 寄存器压力增大：更大的 `tile_size` 可能需要更多的寄存器来保存中间计算结果。
  如果寄存器使用量超过限制，可能导致寄存器溢出到本地内存（位于全局内存上），使得访问延迟增加，性能下降。
- 线程调度开销：增大 `tile_size` 会增加单个线程块的计算时间，导致线程块之间的负载不均衡。
  同时，当线程块数量减少时，线程调度的灵活性降低，可能增加同步等待时间。
- 共享内存访问冲突：大的 `tile_size` 下，共享内存的访问冲突（如银行冲突）可能增加，导致共享内存的访问效率降低，影响性能。


## lab5

### 实现流程

```c
const int BLOCK_SIZE = TILE_WIDTH;
__global__ void MatrixMulSharedMemKernel(float *A, float *B, float *C, int wA,
                                         int wB) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Index of the first sub-matrix of A processed by the block
  int aBegin = wA * BLOCK_SIZE * by;

  // Index of the last sub-matrix of A processed by the block
  int aEnd = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  int aStep = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  int bBegin = BLOCK_SIZE * bx;

  // Step size used to iterate through the sub-matrices of B
  int bStep = BLOCK_SIZE * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin; a <= aEnd; // Ensure all tiles are covered
       a += aStep, b += bStep) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load the matrices from device memory
    // to shared memory; each **thread** loads
    // one element of each matrix
    // --- TO DO :Load the elements of the sub-matrix of A into As ---
    int aRow = a / wA + ty; // Calculate row in A
    int aCol = a % wA + tx; // Calculate column in A
    if (aRow < wA && aCol < wA)
      As[ty][tx] = A[aRow * wA + aCol];
    else
      As[ty][tx] = 0.0f;

    // ---        Load the elements of the sub-matrix of B into Bs ---
    int bRow = b / wB + ty; // Calculate row in B
    int bCol = b % wB + tx; // Calculate column in B
    if (bRow < wA && bCol < wB)
      Bs[ty][tx] = B[bRow * wB + bCol];
    else
      Bs[ty][tx] = 0.0f;

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll
    // --- TO DO :Implement the matrix multiplication using the sub-matrices As
    // and Bs ---
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * Bs[k][tx];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  // --- TO DO :Store the computed Csub result into matrix C ---
  int row_C = by * BLOCK_SIZE + ty;
  int col_C = bx * BLOCK_SIZE + tx;
  if (row_C < wA && col_C < wB) {
    C[c + ty * wB + tx] = Csub;
  }
}

int main(int argc, char * argv[]) {
  // ...
  for (int j = 0; j < nIter; j++) {
    MatrixMulSharedMemKernel<<<grid, block>>>(d_M, d_N, d_P, m, n);
  }
  // ...
}
```

### 测试结果

- 编译运行

```sh
# command
lenovo@10:~/comp-arch-hitsz$ ./a.out 1 1000

# output
Kernel Elpased Time: 43.915 ms
Performance= 45.54 GFlop/s, Time= 43.915 msec, Size= 2000000000 Ops
Computing result using host CPU...done.
Listing first 100 Differences > 0.000010...
 
  Total Errors = 0
```
- 写脚本, 测试运行

```sh
function compile() {
    local use_cublas=$1
    if [ $use_cublas -eq 1 ]; then
        echo "use cublas"
        nvcc -arch=compute_35 -L/usr/local/cuda/lib64 -lcublas ./matrix_mul.cu -o ./a.out
    else
        echo "use no cublas"
        nvcc -arch=compute_35 -L/usr/local/cuda/lib64 ./matrix_mul.cu -o ./a.out
    fi
}

function test() {
    for matrix_size in 1000 3000 5000 10000 20000 ; do
        echo "matrix_size: $matrix_size"
        ./a.out 0 $matrix_size
    done
}

compile 1
test

compile 0
test

```

下面是输出结果

```sh
# command
lenovo@10:~/comp-arch-hitsz$ ./bench5.sh 

# output
matrix_size: 1000
Kernel Elpased Time: 43.741 ms
Performance= 45.72 GFlop/s, Time= 43.741 msec, Size= 2000000000 Ops
matrix_size: 3000
Kernel Elpased Time: 1126.398 ms
Performance= 47.94 GFlop/s, Time= 1126.398 msec, Size= 54000000000 Ops
matrix_size: 5000
Kernel Elpased Time: 5276.184 ms
Performance= 47.38 GFlop/s, Time= 5276.184 msec, Size= 250000000000 Ops
matrix_size: 10000
Kernel Elpased Time: 41417.367 ms
Performance= 48.29 GFlop/s, Time= 41417.367 msec, Size= 2000000000000 Ops
matrix_size: 20000
Kernel Elpased Time: -0.000 ms # error
Performance= -39230013968948910454590928611966976.00 GFlop/s, Time= -0.000 msec, Size= 16000000000000 Ops

# 发现加了算子没区别 ?
```

### 结论

- MatrixMulSharedMemKernel 比 MatrixMulKernel 表现要好，而且随着矩阵变大性能下降并不明显。
