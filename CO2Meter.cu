/*********************************************
UNIVERSIDAD DEL VALLE DE GUATEMALA
CC3056 - Programación de Microprocesadores
Ciclo 2 - 2019

Authors: Gustavo Mendez, Roberto Figueroa, Marco Fuentes
Date: Oct. 27, 2019
File: CO2Meter.cu
Description: Test of measurement model of CO2 in a closed environment, depending on the temperature,
            pressure, volume of room and number of people in it.
**********************************************/
#include <stdio.h> 
#include <stdlib.h> 
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
//CSV Reader 
#include <iostream>
#include <fstream>
using namespace std;

#define N (83500) //Threads, and length of arrays declared
#define THREADS_PER_BLOCK (240) //Threads per block 
#define N_BLOCKS (N/THREADS_PER_BLOCK)

/* Function to load data from CSV file */
void loadGasParams (double *vector1, double *vector2, double *vector3, double *vector4, int n)
{
    //Loading CSV
	ifstream ip("finalData.csv");

	if (!ip.is_open())
		std::cout << "ERROR: File Open" << '\n';

	//Gas means CO2 value, in PPM
	string gasString, temperatureString, pressureString;

    int i = 0;
	while (ip.good())
	{
		//Reading by columns
		getline(ip, gasString, ',');
		getline(ip, temperatureString, ',');
		getline(ip, pressureString, '\n');

		//Cast string to double, and saving values in array
		vector1[i] = ::atof(gasString.c_str());
		vector2[i] = ::atof(temperatureString.c_str());
        vector3[i] = ::atof(pressureString.c_str());
        vector4[i] = 0.0;
        i++;
				
    }
    
    printf("# of data: %d \n", i - 1);

	ip.close();
    return;
}

/* CO2 model check function */
__global__ void gasMeterModelCalc(double *a, double *b, double *c, double *e, double *average)
{
    //__shared__ int productTempVector[THREADS_PER_BLOCK]; //All threads in a block must be able to access this array

    int index = threadIdx.x + blockIdx.x * blockDim.x; //index
    //productTempVector[threadIdx.x] = a[index] * b[index];

    if(index==0) *average = 0; 
    __syncthreads();    

    if( 0 == threadIdx.x ) //Every block to do average += sum
    {
        /*double sum = 0.0;
        for(int j=0; j < THREADS_PER_BLOCK; j++) sum += productTempVector[j];
        atomicAdd(average, sum);*/
    }
}

int main(void)
{
    /* Vectors contains:
        - a: real CO2 value, in PPM
        - b: temperature value, in Celsius
        - c: pressure value, in hPa
        - e: CO2 model value, in PPM
    */

    double *a, *b, *c, *e, *gasAverage; //host copies of a,b,c vectors
    double *d_a, *d_b, *d_c, *d_e, *d_gasAverage; //device copies of a,b,c etc

    int size = N * sizeof(double); //size of memory that needs to be allocated

    //Allocate space for device copies of a,b,c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    cudaMalloc((void **)&d_e, size);

    //Setup input values
    a = (double *)malloc(size);
    b = (double *)malloc(size);
    c = (double *)malloc(size);
    e = (double *)malloc(size);
    loadGasParams(a,b,c,e, N);

    //Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_e, e, size, cudaMemcpyHostToDevice);


    //Gas AVG
    gasAverage = (double *)malloc(sizeof(double)); //Allocate host memory to gasAverage
    *gasAverage = 0.0; 
    cudaMalloc((void **)&d_gasAverage, sizeof(double)); //Allocate device memory to d_gasAverage

    

    //Timing
    struct timeval t1, t2;
    gettimeofday(&t1, 0);
    
    /*
    gasMeterModelCalc<<<N_BLOCKS,THREADS_PER_BLOCK>>>(d_a, d_b, d_c, d_e, d_gasAverage); 
    cudaMemcpy(dotProduct, d_dotProduct, sizeof(int), cudaMemcpyDeviceToHost); //Copy result into gasAverage
    printf("\n CO2 AVG: %d\n", *gasAverage); //Output result
    */

    gettimeofday(&t2, 0);
    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    printf("TIME FOR 240 THREADS / 1 BLOCKS:  %3.3f ms \n", time);

    //Clean
    free(a); 
    free(b); 
    free(c); 
    free(e);
    free(gasAverage);

    //Cuda free
    cudaFree(d_a); 
    cudaFree(d_b); 
    cudaFree(d_c); 
    cudaFree(d_e);
    cudaFree(d_gasAverage);

    return 0;
} 