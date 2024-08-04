#include <stdio.h>
#include <string.h>
//#include <omp.h>

#define N 2048
#define T 64

//int ncpus;
float a[N][N], b[N][N], c[N][N];

void mat_mul(){
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < N; k++){
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

int main(){

    //cpu_set_t cpu_set;
    //sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    //ncpus = CPU_COUNT(&cpu_set);

    FILE *fa = fopen("a.dat", "rb");
    FILE *fb = fopen("b.dat", "rb");

    fread_unlocked(a, sizeof(float), sizeof(a) / sizeof(float), fa);
    fread_unlocked(b, sizeof(float), sizeof(b) / sizeof(float), fb);
    fclose(fa);
    fclose(fb);

    memset(c, 0, sizeof(c));
    mat_mul();

    FILE *fc = fopen("c.dat", "wb");
    fwrite(c, sizeof(float), sizeof(c) / sizeof(float), fc);
    fclose(fc);

    return 0;
}
