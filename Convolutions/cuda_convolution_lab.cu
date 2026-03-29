#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <vector>

using namespace std;

/*
    CUDA Convolution Lab
    ====================
    Implements 1D and 2D full convolution on the GPU.

    Convolution Type: Full
        Output size = input size + kernel size - 1
        Requires padding of size (kernel_size - 1) on each side

    Padding Strategy: Explicit (host-side)
        A padded copy of the input is created on the host before launching the kernel.
        This avoids bounds checks inside the kernel (no branch divergence).
        Tradeoff: small extra allocation + memcpy on host vs cleaner, divergence-free kernel.

    Tiled Version (2D):
        Uses shared memory to reduce global memory traffic.
        Each block cooperatively loads a tile of the input (including halo) into shared memory.
        The convolution filter is stored in __constant__ memory — broadcast read to all threads in a warp.
        Halo elements that fall outside the input are implicitly zero padded via bounds check during load.
*/


// -------------------------------------------------------------------------------------------------
// 1D CONVOLUTION
// -------------------------------------------------------------------------------------------------

__global__
void cuda_conv1d_full(const float* I_padded, const float* K, float* C,
                      const int k_width, const int c_width) {
    // Each thread is responsible for exactly one output element
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard — more threads may be launched than output elements
    if (c_idx >= c_width) return;

    // Slide kernel over padded input — no bounds check needed (explicit padding)
    // I_padded has (kernel_width - 1) zeros on each side
    // so c_idx + k is always a valid index
    float val = 0.f;
    for (int k = 0; k < k_width; ++k)
        val += I_padded[c_idx + k] * K[k];

    C[c_idx] = val;
}

__host__
void conv1d_full(const vector<float>& I, const vector<float>& K, vector<float>& C) {
    // Pad input — full conv padding = kernel_size - 1 on each side
    const int pad_width     = K.size() - 1;
    const int i_padded_width = I.size() + pad_width * 2;
    vector<float> I_padded(i_padded_width, 0.f);
    copy(I.begin(), I.end(), I_padded.begin() + pad_width);

    const int    c_width       = I.size() + K.size() - 1;
    const size_t i_padded_size = i_padded_width  * sizeof(float);
    const size_t k_size        = K.size()         * sizeof(float);
    const size_t c_size        = c_width          * sizeof(float);

    float *d_I, *d_K, *d_C;
    cudaMalloc(&d_I, i_padded_size);
    cudaMalloc(&d_K, k_size);
    cudaMalloc(&d_C, c_size);

    cudaMemcpy(d_I, I_padded.data(), i_padded_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K.data(),        k_size,        cudaMemcpyHostToDevice);

    dim3 blockDim(32);
    dim3 gridDim((c_width + 31) / 32);

    cuda_conv1d_full<<<gridDim, blockDim>>>(d_I, d_K, d_C, K.size(), c_width);
    cudaDeviceSynchronize();

    cudaMemcpy(C.data(), d_C, c_size, cudaMemcpyDeviceToHost);

    cudaFree(d_I);
    cudaFree(d_K);
    cudaFree(d_C);
}


// -------------------------------------------------------------------------------------------------
// 2D CONVOLUTION
// -------------------------------------------------------------------------------------------------

__global__
void cuda_conv2d_full(const float* I_padded, const float* K, float* C,
                      const int k_h, const int k_w,
                      const int c_h, const int c_w,
                      const int i_padded_w) {
    // Each thread owns one output element
    int c_col = blockIdx.x * blockDim.x + threadIdx.x;
    int c_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (c_col >= c_w || c_row >= c_h) return;

    // Slide 2D kernel over padded input — no bounds check needed (explicit padding)
    float val = 0.f;
    for (int k_row = 0; k_row < k_h; ++k_row) {
        int i_row = c_row + k_row;
        for (int k_col = 0; k_col < k_w; ++k_col)
            val += I_padded[i_row * i_padded_w + c_col + k_col] * K[k_row * k_w + k_col];
    }

    C[c_row * c_w + c_col] = val;
}

__host__
void conv2d_full(const vector<float>& I, const vector<float>& K, vector<float>& C,
                 const int i_h, const int i_w,
                 const int k_h, const int k_w) {
    // Pad in both dimensions — full conv padding = kernel_size - 1 on each side
    const int pad_h      = k_h - 1;
    const int pad_w      = k_w - 1;
    const int i_padded_h = i_h + pad_h * 2;
    const int i_padded_w = i_w + pad_w * 2;

    // Build padded input — row by row copy into padded grid
    vector<float> I_padded(i_padded_h * i_padded_w, 0.f);
    for (int r = 0; r < i_h; ++r)
        copy(I.begin() + r * i_w,
             I.begin() + r * i_w + i_w,
             I_padded.begin() + (r + pad_h) * i_padded_w + pad_w);

    const int c_h = i_h + k_h - 1;
    const int c_w = i_w + k_w - 1;

    const size_t i_padded_size = i_padded_h * i_padded_w * sizeof(float);
    const size_t k_size        = k_h * k_w               * sizeof(float);
    const size_t c_size        = c_h * c_w               * sizeof(float);

    float *d_I, *d_K, *d_C;
    cudaMalloc(&d_I, i_padded_size);
    cudaMalloc(&d_K, k_size);
    cudaMalloc(&d_C, c_size);

    cudaMemcpy(d_I, I_padded.data(), i_padded_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K.data(),        k_size,        cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((c_w + 15) / 16, (c_h + 15) / 16);

    cuda_conv2d_full<<<gridDim, blockDim>>>(d_I, d_K, d_C, k_h, k_w, c_h, c_w, i_padded_w);
    cudaDeviceSynchronize();

    cudaMemcpy(C.data(), d_C, c_size, cudaMemcpyDeviceToHost);

    cudaFree(d_I);
    cudaFree(d_K);
    cudaFree(d_C);
}


// -------------------------------------------------------------------------------------------------
// 2D CONVOLUTION — TILED (shared memory + constant filter)
// -------------------------------------------------------------------------------------------------
//
//  Optimization over naive 2D conv:
//    - Filter stored in __constant__ memory — single broadcast read per warp instead of 32 global reads
//    - Input tile loaded cooperatively into shared memory — each element read from global memory once
//    - Halo elements (border of tile needed by edge threads) loaded with implicit zero padding
//
//  Padding strategy: implicit (bounds check during shared memory load)
//    - No padded input allocation needed on host
//    - Out of bounds loads write 0.f into shared memory tile (equivalent to zero padding)
//    - Border block warps incur minor divergence during load — interior blocks are divergence-free
//
//  Output: same size as input (same padding)
//    - Border output elements include zero-padded halo contributions

#define KERNEL_RADIUS 1                          // filter radius — 3x3 kernel = radius 1
#define TILE_W        16                         // output tile width
#define TILE_H        16                         // output tile height
#define SM_W          (TILE_W + 2 * KERNEL_RADIUS)  // shared memory tile width  (includes halo)
#define SM_H          (TILE_H + 2 * KERNEL_RADIUS)  // shared memory tile height (includes halo)

// Filter stored in constant memory — read-only, broadcast to all threads in a warp
__constant__ float F[2 * KERNEL_RADIUS + 1][2 * KERNEL_RADIUS + 1];

__global__
void cuda_conv2d_full_tiled(const float* I, float* C, const int i_h, const int i_w) {
    // Shared memory tile — sized to include halo on all sides
    __shared__ float tile[SM_H][SM_W];

    // Global output element this thread owns
    int c_col = blockIdx.x * TILE_W + threadIdx.x;
    int c_row = blockIdx.y * TILE_H + threadIdx.y;

    // Corresponding input element — offset by radius to include halo
    int i_col = c_col - KERNEL_RADIUS;
    int i_row = c_row - KERNEL_RADIUS;

    // Cooperative load — every thread loads one element into shared memory
    // Halo threads (i_row/i_col out of bounds) write 0.f — implicit zero padding
    if (i_row >= 0 && i_row < i_h && i_col >= 0 && i_col < i_w)
        tile[threadIdx.y][threadIdx.x] = I[i_row * i_w + i_col];
    else
        tile[threadIdx.y][threadIdx.x] = 0.f;

    // Wait for all threads to finish loading before computing
    __syncthreads();

    // Guard — halo threads don't own a valid output element
    if (c_row >= i_h || c_col >= i_w) return;

    // Compute dot product entirely from shared memory — no global memory access
    float val = 0.f;
    for (int ky = 0; ky < 2 * KERNEL_RADIUS + 1; ++ky)
        for (int kx = 0; kx < 2 * KERNEL_RADIUS + 1; ++kx)
            val += tile[threadIdx.y + ky][threadIdx.x + kx] * F[ky][kx];

    C[c_row * i_w + c_col] = val;
}

__host__
void conv2d_full_tiled(const float* h_I, const float* h_F, float* h_C,
                       const int i_h, const int i_w) {
    const int    k_dim  = 2 * KERNEL_RADIUS + 1;
    const size_t i_size = i_h * i_w * sizeof(float);
    const size_t c_size = i_h * i_w * sizeof(float);

    // Copy filter to constant memory (not cudaMemcpy — special symbol copy)
    cudaMemcpyToSymbol(F, h_F, k_dim * k_dim * sizeof(float));

    float *d_I, *d_C;
    cudaMalloc(&d_I, i_size);
    cudaMalloc(&d_C, c_size);

    cudaMemcpy(d_I, h_I, i_size, cudaMemcpyHostToDevice);

    // blockDim covers full shared memory tile (output tile + halo)
    // gridDim covers output in steps of TILE_W / TILE_H
    dim3 blockDim(SM_W, SM_H);
    dim3 gridDim((i_w + TILE_W - 1) / TILE_W,
                 (i_h + TILE_H - 1) / TILE_H);

    cuda_conv2d_full_tiled<<<gridDim, blockDim>>>(d_I, d_C, i_h, i_w);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, c_size, cudaMemcpyDeviceToHost);

    cudaFree(d_I);
    cudaFree(d_C);
}


// -------------------------------------------------------------------------------------------------
// MAIN
// -------------------------------------------------------------------------------------------------

int main(void) {

    // 1D Convolution
    // {
    //     constexpr int input_size = 32;
    //     constexpr int kernel_size = 5;
    //     constexpr int conv_size = input_size + kernel_size - 1;
    //     vector<float> I(input_size, 1.f);
    //     vector<float> K(kernel_size, 1.f);
    //     vector<float> C(conv_size, 0.f);
    //     conv1d_full(I, K, C);
    // }

    // 2D Convolution
    // {
    //     constexpr int i_h = 8, i_w = 8;
    //     constexpr int k_h = 3, k_w = 3;
    //     constexpr int c_h = i_h + k_h - 1;
    //     constexpr int c_w = i_w + k_w - 1;
    //     static_assert(k_h % 2 == 1 && k_h == k_w, "kernel must be square and odd");
    //     vector<float> I(i_h * i_w, 1.f);
    //     vector<float> K(k_h * k_w, 1.f);
    //     vector<float> C(c_h * c_w, 0.f);
    //     conv2d_full(I, K, C, i_h, i_w, k_h, k_w);
    // }

    // 2D Tiled Convolution
    {
        constexpr int k_dim = 2 * KERNEL_RADIUS + 1;
        constexpr int i_h = 8, i_w = 8;
        constexpr int c_h = i_h + k_dim - 1;
        constexpr int c_w = i_w + k_dim - 1;
        vector<float> I(i_h * i_w, 1.f);
        vector<float> K(k_dim * k_dim, 1.f);
        vector<float> C(c_h * c_w, 0.f);
        conv2d_full_tiled(I.data(), K.data(), C.data(), i_h, i_w);
    }

    return 0;
}
