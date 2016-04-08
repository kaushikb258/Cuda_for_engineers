#include <iostream>
#include <math.h>
using namespace std;

#define W 500
#define H 500
#define TPB 32

__device__ unsigned char clip(int n)
{
 if (n>255) return n;
 else if (n<0) return 0;
 else return n;
}

__device__ int square(int x)
{
 return (x*x);
}

__global__ void distKernel(uchar4 *dout, int w, int h, int2 pos)
{
 const int c = blockIdx.x*blockDim.x + threadIdx.x;
 const int r = blockIdx.y*blockDim.y + threadIdx.y;
 const int i = r*w + c;
 if ((c >= w) || (r >= h)) return;
 int d = sqrtf(square(c-pos.x) + square(r-pos.y));
 unsigned char intensity = clip(255-d);

 dout[i].x = intensity; // red
 dout[i].y = intensity; // green
 dout[i].z = 0; // blue
 dout[i].w = 255; // opaque
}

int main()
{
 uchar4 *out = new uchar4 [W*H]; 
 uchar4 *dout = new uchar4;
 const int size = W*H*sizeof(uchar4);
 cudaMalloc(&dout,size); 
 const int2 pos = {0, 0};
 const dim3 tpb(TPB, TPB);
 const dim3 bpg((W+TPB-1)/TPB, (H+TPB-1)/TPB);
 
 distKernel<<<bpg,tpb>>>(dout,W,H,pos);
 cudaMemcpy(out,dout,size,cudaMemcpyDeviceToHost);

 cudaFree(dout);
 delete [] out;
}

