#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <mpi.h>
using namespace std;

//TODO: turn it to MPI program!
int main(int argc, char *argv[])
{   
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc <2){
        if (rank == 0){ 
                cerr << "Usage: " << argv[0] << "<number of tests>" << endl;
        }
        MPI_Finalize();
        return -1;
    }
    
    long long total_tests = strtoll(argv[1], NULL, 10);
    long long local_tests = total_tests / size;
    long long leftover_tests = total_tests % size;
    
    if (rank==0){
        local_tests += leftover_tests;
    }
    
    
    long long local_circle_cnt = 0;
    srand(time(NULL) + rank);
    
    for (long long i=0; i < local_tests; ++i){
                double x = (double)rand() / RAND_MAX;
                double y = (double)rand() / RAND_MAX;
                
                double d = x*x + y*y;
                if(d <= 1) local_circle_cnt++;
        }
    
    
    long long global_circle_cnt = 0;
    MPI_Reduce(&local_circle_cnt, &global_circle_cnt,1,MPI_LONG_LONG,MPI_SUM,0,MPI_COMM_WORLD);
        
    MPI_Finalize();
    if (rank ==0){
        double pi = 4.0 * global_circle_cnt / total_tests;
        cout << "Pi = " << pi << endl;
    }
    return 0;
}