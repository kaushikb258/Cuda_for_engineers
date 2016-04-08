#include <iostream>
#include <stdio.h>
#define N 64
#define TPB 32


float scale(int i, int n)
{
 return ((float) i)/((float) (n-1));
}

__device__ float distance(float x1, float x2)
{
 float x = (x1-x2)*(x1-x2);
 return sqrt(x);
}

__global__ void distanceKernel(float *d_out, float *d_in, float ref)
{
 const int i = blockIdx.x*blockDim.x + threadIdx.x;
 const float x = d_in[i];
 d_out[i] = distance(x,ref);
 printf("out [ %d ] = %f \n", i, d_out[i]);
}

int main()
{
 float *in = new float;
 float *out = new float;
 const float ref = 0.5;  

 cudaMallocManaged(&in, N*sizeof(float));
 cudaMallocManaged(&out, N*sizeof(float));
 
 for (int i=0; i<N; i++)
 {
  in[i] = scale(i,N);
 }

 distanceKernel<<<N/TPB,TPB>>>(out, in, ref);
 cudaDeviceSynchronize();

 cudaFree(in);
 cudaFree(out);  
}
