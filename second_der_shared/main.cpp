#include <iostream>
#include <math.h>
#include "kernel.h"
#include "global.h"
#include <stdio.h>
#include <fstream>
using namespace std;

int main()
{
 float u[N];
 float out[N];
 float i_by_N;

 for (int i=0; i<N; i++)
 {
  i_by_N = ((float) i)/((float) N);
  u[i] = sin(2.0*PI*i_by_N);
 }

 func(u,out);

 ofstream f;
 f.open("output",ios_base::out);
 for (int i=0; i<N; i++)
 {
  f<<i<<" "<<u[i]<<" "<<out[i]+u[i]<<endl;
 }
 f.close();
}
