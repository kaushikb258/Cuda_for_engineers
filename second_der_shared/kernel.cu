#include <iostream>
#include <stdio.h>
#include "kernel.h"
#include "global.h"
using namespace std;

__device__ float compDer(float xim1, float xi, float xip1)
{
 return ((xim1 - 2.0*xi + xip1)/(h*h));
}

__global__ void derivative(float  *d_in, float *d_out)
{
 const int idx = blockIdx.x*blockDim.x + threadIdx.x;
 if (idx>N) return;
 
 extern __shared__ float s_in[];
 
 // interior cells
 s_in[threadIdx.x+1] = d_in[idx]; 

 // ghost cells
 if(blockIdx.x != 0 && threadIdx.x==0) s_in[0] = d_in[idx-1];  
 if(blockIdx.x != gridDim.x-1 && threadIdx.x==blockDim.x-1) s_in[blockDim.x+1] = d_in[idx+1]; 

 // boundary cells, use periodic BC
 if (blockIdx.x==0 && threadIdx.x==0) s_in[0] = d_in[N-1];  
 if (blockIdx.x==gridDim.x-1 && threadIdx.x==blockDim.x-1) s_in[blockDim.x+1] = d_in[0];

 __syncthreads();
 
 d_out[idx] = compDer(s_in[threadIdx.x],s_in[threadIdx.x+1],s_in[threadIdx.x+2]); 
}

void func(float *in, float *out)
{
 float *d_in = 0;
 float *d_out = 0;
 int size = N*sizeof(float);
 const int BPG = (N + TPB -1)/TPB;
 cout<<"BPG = "<<BPG<<endl;
 cout<<"TPB = "<<TPB<<endl;
 const int s_size = (TPB+2)*sizeof(float);

 cudaMalloc(&d_in,size); 
 cudaMalloc(&d_out,size);  

 cudaMemcpy(d_in,in,size,cudaMemcpyHostToDevice);
 derivative<<<BPG,TPB,s_size>>>(d_in,d_out);
 cudaMemcpy(out,d_out,size,cudaMemcpyDeviceToHost);

 cudaFree(d_in);
 cudaFree(d_out);
}
