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
