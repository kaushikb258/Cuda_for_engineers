#include <iostream>
#include <math.h>
using namespace std;

#define W 500
#define H 500
#define TPB 32

__device__ float square(float x)
{
 return (x*x);
}

__global__ void distKernel(float *dout, int w, int h, float2 pos)
{
 const int c = blockIdx.x*blockDim.x + threadIdx.x;
 const int r = blockIdx.y*blockDim.y + threadIdx.y;
 const int i = r*w + c;
 if ((c >= w) || (r >= h)) return;
 dout[i] = sqrt(square(c-pos.x) + square(r-pos.y));
}

int main()
{
 float *out = new float [W*H]; 
 float *dout = new float;
 const int size = W*H*sizeof(float);
 cudaMalloc(&dout,size); 
 const float2 pos = {0.0, 0.0};
 const dim3 tpb(TPB, TPB);
 const dim3 bpg((W+TPB-1)/TPB, (H+TPB-1)/TPB);
 
 distKernel<<<bpg,tpb>>>(dout,W,H,pos);
 cudaMemcpy(out,dout,size,cudaMemcpyDeviceToHost);

 cudaFree(dout);
 delete [] out;
}

