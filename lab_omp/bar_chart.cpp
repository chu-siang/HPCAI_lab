#include <iostream>
#include <omp.h>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cout << "Error: No input file provided." << std::endl;
        return 1;
    }
    char *inputfile_name = argv[1];

    if (argc < 3)
    {
        std::cout << "Error: No output file provided." << std::endl;
        return 1;
    }
    char *outputfile_name = argv[2];

    if (freopen(inputfile_name, "r", stdin) == nullptr)
    {
        std::cout << "Error: Unable to open input file: " << inputfile_name << std::endl;
        return 1;
    }

    std::cout << "Reading input file..." << std::endl;

    int n = 0;

    std::cin >> n;

    int **array = (int **)malloc(sizeof(int *) * n);
    
        for (int i = 0; i < n; i++)
        {
            array[i] = (int *)malloc(sizeof(int) * 5);
            for (int j = 0; j < 5; j++)
            {
                std::cin >> array[i][j];
            }
        }
    
    int grade_span[5][5];
    #pragma omp parallel for
    for (int i = 0; i < 5; i++)
    {
        for(int j = 0; j < 5; j++){
            grade_span[i][j] = 0;
        }
    }

    std::cout << "Finish reading input file. Total " << n << " records." << std::endl;
    std::cout << "Start sorting..." << std::endl;
    auto start = std::chrono::steady_clock::now();
    #pragma omp parallel for reduction(+:grade_span[0:5][0:5])
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (array[j][i] <= 100 && array[j][i] >= 90)
                {
                    grade_span[i][0]++;
                }
                else if (array[j][i] < 90 && array[j][i] >= 80)
                {
                    grade_span[i][1]++;
                }
                else if (array[j][i] < 80 && array[j][i] >= 70)
                {
                    grade_span[i][2]++;
                }
                else if (array[j][i] < 70 && array[j][i] >= 60)
                {
                    grade_span[i][3]++;
                }
                else if (array[j][i] < 60 && array[j][i] >= 0)
                {
                    grade_span[i][4]++;
                }
                else printf("error");
            }
        }
    auto end = std::chrono::steady_clock::now();
    std::cout << "Sorting complete." << std::endl;
    std::cout << "Sorting Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    std::cout << "Start plotting..." << std::endl;

    for(int i=0;i<5;i++){
        std::vector<double> x = {0+0.8*i, 5+0.8*i, 10+0.8*i, 15+0.8*i, 20+0.8*i};
        std::vector<double> h = {(double)grade_span[i][0], (double)grade_span[i][1], (double)grade_span[i][2], (double)grade_span[i][3], (double)grade_span[i][4]};
        plt::bar(x, h);   
    }
    std::vector<double> x_positions = {1.6, 6.6, 11.6, 16.6, 21.6};
    std::vector<std::string> x_labels = {"100-90", "90-80", "80-70", "70-60", "below 60"};
    plt::xticks(x_positions, x_labels);
    plt::save(outputfile_name);
    std::cout << "Plotting complete." << std::endl;
    std::cout << "Saved as " << outputfile_name << std::endl;

    return 0;
}
