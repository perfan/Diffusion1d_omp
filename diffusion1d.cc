// 
// diffusion1d.cc
//
// 1d diffusion with periodic boundary conditions
//
// Compile with make using provided Makefile 
//

#include <fstream>
#include <rarray>
#include "diffusion1d_output.h"
#include "diffusion1d_timestep.h"
#include "parameters.h"
#include <omp.h>
#include <iostream>

// the main function drives the simulation
int main(int argc, char *argv[]) 
{
  
  // Simulation parameters
  double      L;  // system length
  double      D;  // diffusion constant
  double      T;  // time
  double      dx; // spatial resolution
  double      dt; // temporal resolution (time step)
  int         Z;  // number of walkers (dummy variable, not used)
  std::string datafile; // filename for output
  double      time_between_output;

  // Read parameters from a file given on the command line. 
  // If no file was given, use "params.ini".
  std::string paramFilename = argc>1?argv[1]:"params.ini";
  read_parameters(paramFilename, L, D, T, dx, dt, Z, datafile, time_between_output);

  // Compute derived parameters 
  const int numSteps = int(T/dt + 0.5);  // number of steps to take
  const int N = int(L/dx + 0.5);         // number of grid points
  const int outputEvery = int(time_between_output/dt + 0.5); // how many steps between output
  const int outputcols = 48;             // number of columns for sparkline output
  int tid;                               // thead id for the parallel parts of the code
  int numthreads;                        // total number of activated threads
  
  // Allocate density data 
  rvector<double> P(N);

  // Setup initial conditions for P
  P.fill(0.0);
  P[N/2] = 1.0;

  // Setup initial time
  double time = 0.0;    

  // Open a file for data output
  std::ofstream file;
  diffusion1d_output_init(file, datafile);

  // Initial output
  diffusion1d_output(file, 0, time, P, outputcols);

// "Forking" the threads for parallel processing
#pragma omp parallel private(tid) shared(numthreads)    
{ 
  numthreads = omp_get_num_threads();                   // gets the total number of activated threads
  tid = omp_get_thread_num();                           // gets the id of each thread
  
  // Time evolution
  for (int step = 1; step <= numSteps; step++) {

    // Compute next time point
    diffusion1d_timestep(P, D, dt, dx);

    // This section needs to be done only by only one (master) to avoid any duplication
    #pragma omp master                                   
    {
    // Update time
    time += dt;

    // Periodically add data to the file
    if (step % outputEvery == 0 and step > 0) 
        diffusion1d_output(file, step, time, P, outputcols);
    }

  }
}  
  // Close file
  diffusion1d_output_finish(file);

  // All done
  return 0;
}
  
