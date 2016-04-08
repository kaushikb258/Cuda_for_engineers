#include <iostream>
#include <stdio.h>
#include "kernel.h"
#define N 1024
using namespace std;

int main()
{
 float *a = new float [N];
 float *b = new float [N];
 float *res = new float;

 for (int i=0; i<N; i++)
 {
  a[i] = 1.0;
  b[i] = 2.0;
 }

 dot(res,a,b,N);
 cout<<"result = "<<*res<<endl;

 delete [] a;
 delete [] b;
 delete res;
}
