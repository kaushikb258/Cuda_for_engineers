#include <iostream>
#include "kernel.h"
#define TPB 64
#define VERSION 2

__global__ void dotKernel(float *d_res, const float *d_a, const float *d_b, const int n)
{
 const int idx = threadIdx.x + blockIdx.x*blockDim.x;
 if(idx>n) return;

 __shared__ float s_prod[TPB];
 const int s_idx = threadIdx.x;

 s_prod[s_idx] = d_a[idx]*d_b[idx];
 __syncthreads();


 if (VERSION == 1)
 { 
  if (s_idx == 0)
  {
   float blockSum = 0;
   for (int j=0; j<blockDim.x; j++)
   {
    blockSum += s_prod[j];
   }
   atomicAdd(d_res,blockSum);
  }
 }
 else if (VERSION == 2)
 {

  // reduction
  for (int i = blockDim.x/2; i>0; i /= 2)
  {
   if (s_idx < i) s_prod[s_idx] += s_prod[s_idx + i];
   __syncthreads();
  }
  
  if(s_idx ==0) atomicAdd(d_res,s_prod[s_idx]);
 }
 

}


void dot(float *res, const float *a, const float *b, const int n)
{
 float *d_res;
 float *d_a = 0;
 float *d_b = 0;

 cudaMalloc(&d_res,sizeof(float));
 cudaMalloc(&d_a,n*sizeof(float));
 cudaMalloc(&d_b,n*sizeof(float));

 cudaMemset(d_res,0,sizeof(float));
 cudaMemcpy(d_a,a,n*sizeof(float),cudaMemcpyHostToDevice);
 cudaMemcpy(d_b,b,n*sizeof(float),cudaMemcpyHostToDevice);

 dotKernel<<<(n+TPB-1)/TPB,TPB>>>(d_res, d_a, d_b, n);
 cudaMemcpy(res,d_res,sizeof(float),cudaMemcpyDeviceToHost);

 cudaFree(d_res);
 cudaFree(d_a);
 cudaFree(d_b);
}
