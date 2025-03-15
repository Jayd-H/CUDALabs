/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _BICUBICTEXTURE_CU_
#define _BICUBICTEXTURE_CU_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <helper_math.h>

 // includes, cuda
#include <helper_cuda.h>

typedef unsigned int uint;
typedef unsigned char uchar;


cudaArray* d_imageArray = 0;


__global__ void d_render(uchar4* d_output, uint width, uint height) {
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint i = __umul24(y, width) + x;
    if ((x < width) && (y < height)) {

        // normalise to 1, -1
        float u = x / (float)width;
        float v = y / (float)height;
        u = 2.0f * u - 1.0f;
        v = -(2.0f * v - 1.0f);

        // scale coords to 2, -2 to zoom in rather than 4, -4
        u *= 2.0f;
        v *= 2.0f;

        // julia set
        float2 z = make_float2(u, v);

        // For Julia set, c is fixed rather than being the starting point
        float2 c = make_float2(0.25f, 0.5f);

        float r = 0.0f;
        float color = 1.0f;

        // iterate 30 times (given by the lab pseudocode)
        for (int j = 0; j < 30; j++) {
            // z = z^2 + c
            float2 z_squared = make_float2(z.x * z.x - z.y * z.y, 2.0f * z.x * z.y);
            z = make_float2(z_squared.x + c.x, z_squared.y + c.y);

            // calculate magnitude of z
            r = sqrtf(z.x * z.x + z.y * z.y);

            // if magnitude of z is greater than 5, the pixel is not in the set
            if (r > 5.0f) {
                color = 0.0f;
                break;
            }
        }
        // set pixel colour based on if its in the set or not
        if (color > 0.0f) {
            // pixels in the set are red
            d_output[i] = make_uchar4(0, 0, 0xff, 0);
        }
        else {
            // pixels that are not are in black
            d_output[i] = make_uchar4(0, 0, 0, 0);
        }
    }
}


extern "C" void freeTexture() {

    checkCudaErrors(cudaFreeArray(d_imageArray));
}

// render image using CUDA
extern "C" void render(int width, int height,  dim3 blockSize, dim3 gridSize,
     uchar4 * output) {


            d_render << <gridSize, blockSize >> > (output, width, height);


    getLastCudaError("kernel failed");
}

#endif