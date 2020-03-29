// 
// diffusion1d_timestep.cc
//
// Time step module for 1d diffusion with periodic boundary conditions 
//

#include "diffusion1d_timestep.h"
#include <omp.h>
#include <iostream>
#include "ticktock.h"

// perform a single time step
void diffusion1d_timestep(rvector<double>& P, double D, double dt, double dx)
{    
   static rvector<double> laplacian;
   const int N = P.size();
   if (laplacian.size() != N) laplacian = rvector<double>(N);

   const double alpha = D*dt/(dx*dx);

   // This section should be done only by one thread to avoid over doing it
   #pragma omp master   
   laplacian[0] = P[N-1] + P[1] - 2*P[0];

   // Parllel for for speeding-up
   #pragma omp for 
   for (int i = 1; i < laplacian.size()-1; i++){
      laplacian[i] = P[i-1] + P[i+1] - 2*P[i];
   }

   // This section should be done only by one thread to avoid over doing it
   #pragma omp master
   laplacian[N-1] = P[N-2] + P[0] - 2*P[N-1];
   
   // Parllel for for speeding-up
   #pragma omp for
   for (int i = 0; i < laplacian.size(); i++)
      P[i] += alpha*laplacian[i];  
}
