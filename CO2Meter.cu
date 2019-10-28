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

#define N 83000                 //Threads, and length of arrays declared

#define R (8.314f)               //Gas constant
#define ACH (3.5f)               //ACH for a classroom
#define CO2_MASS (44.01f)        //CO2 Molar mass 
#define CO2_ADULT_GAIN (0.0052f) //CO2 gain by an adult, in l/s
#define A304_VOLUME (114.20f)    //m3
#define C114_VOLUME (125.625f)   //m3
#define A304_PERSONS (22.0f)       // no. persons in this classroom
#define C114_PERSONS (16.0f)       // no. persons in this classroom

#define CO2_OUTDOOR (404.0f)       //CO2 outside, in PPM
#define PRESSURE_OUTDOOR (852.0f)  //Pressure outside
#define TEMP_OUTDOOR (24.1f)     //Temperature outside 

/* Function to load data from CSV file */
void loadGasParams (float *a, float *b, float *c, int n)
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

        //Cast string to float, and saving values in array
		double gasDouble = ::atof(gasString.c_str());
		double tempDouble = ::atof(temperatureString.c_str());
        double pressureDouble = ::atof(pressureString.c_str());

        a[i] = float(gasDouble);
        b[i] = float(tempDouble);
        c[i] = float(pressureDouble);

        i++;
				
    }
    
    printf("# OF DATA: %d \n", i - 1);

	ip.close();
    return;
}

/* CO2 model check function */
__global__ void gasMeterModelCalc(float *a, float *b, float *c, float *e)
{
    
    //AVG a[index]
    float currVolume = (N/2 > 41500) ? C114_VOLUME : A304_VOLUME;
    float currPersons = (N/2 > 41500) ? C114_PERSONS : A304_PERSONS;

    int index = threadIdx.x + blockDim.x * blockIdx.x;				
	if (index < N)
	{
		e[index] = (b[index] / (c[index] * (1 + ACH * (index/10)))) * ( ((a[0] * c[0])/b[0]) + ( (((CO2_ADULT_GAIN * currPersons * R)/(CO2_MASS * currVolume)) + ((CO2_OUTDOOR * PRESSURE_OUTDOOR * ACH)/(TEMP_OUTDOOR))) * (index/10)));
	}
    
}

int main(int argc, char** argv)
{
    /* Vectors contains:
        - a: real CO2 value, in PPM
        - b: temperature value, in Celsius
        - c: pressure value, in hPa
        - e: CO2 model value, in PPM
    */

    cudaStream_t myStream;
    cudaStreamCreate(&myStream);
    
    float *a, *b, *c, *e; //host copies of a,b,c vectors
    float *d_a, *d_b, *d_c, *d_e; //device copies of a,b,c etc

    int size = N * sizeof(float); //size of memory that needs to be allocated

    //Allocate space for device copies of a,b,c, e
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    cudaMalloc((void **)&d_e, size);

    //Allows device to get access to memory
    cudaHostAlloc( (void**)&a, N * sizeof(int), cudaHostAllocDefault);	 
    cudaHostAlloc( (void**)&b, N * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc( (void**)&c, N * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc( (void**)&e, N * sizeof(int), cudaHostAllocDefault);

    //loading arrays
    loadGasParams(a, b, c, N);

    //Timing
    struct timeval t1, t2;
    gettimeofday(&t1, 0);

    //Using streams
    for(int i=0; i < N; i+= N*2) { 
        // using stream 1 and steam 2
        cudaMemcpyAsync(d_a, a, N*sizeof(int), cudaMemcpyHostToDevice, myStream);
        cudaMemcpyAsync(d_b, b, N*sizeof(int), cudaMemcpyHostToDevice, myStream);
        cudaMemcpyAsync(d_c, c, N*sizeof(int), cudaMemcpyHostToDevice, myStream);
        
        
        gasMeterModelCalc<<<(int)ceil(N/1024)+1, 1024, 0, myStream>>>(d_a, d_b, d_c, d_e);
        cudaMemcpyAsync(e, d_e, N*sizeof(int), cudaMemcpyDeviceToHost, myStream);
    }

    gettimeofday(&t2, 0);
    float time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    printf("EXECUTION TIME:  %5.4f ms \n", time);


    int i;
    //Some values of real CO2 from 3000 to 3010
    printf("MODEL CO2 = [");
    for (i=25000; i<25010; i++) printf(" %4.3f", e[i]);
    printf(" ...]\n");

    //Some values of real CO2 from 3000 to 3010
    printf("REAL CO2 = [");
    for (i=25000; i<25010; i++) printf(" %4.3f", a[i]);
    printf(" ...]\n");

    
    //Destroying stream used
    cudaStreamDestroy(myStream);
	return 0;
} 