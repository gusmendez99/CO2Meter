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

/* CO2 model check function */
__global__ void getGasModelKernel1(float *a1, float *b1, float *c1, float *d1)
{
    float currVolume = A304_VOLUME;
    float currPersons = A304_PERSONS;

    int index = threadIdx.x + blockDim.x * blockIdx.x;				
	if (index < N)
	{
		d1[index] = (b1[index] / (c1[index] * (1 + ACH * (index/10)))) * ( ((a1[0] * c1[0])/b1[0]) + ( (((CO2_ADULT_GAIN * currPersons * R)/(CO2_MASS * currVolume)) + ((CO2_OUTDOOR * PRESSURE_OUTDOOR * ACH)/(TEMP_OUTDOOR))) * (index/10)));
	}
    
}

/* CO2 model check function */
__global__ void getGasModelKernel2(float *a2, float *b2, float *c2, float *d2)
{    
    //AVG a[index]
    float currVolume = C114_VOLUME;
    float currPersons = C114_PERSONS;

    int index = threadIdx.x + blockDim.x * blockIdx.x;				
	if (index < N)
	{
		d2[index] = (b2[index] / (c2[index] * (1 + ACH * (index/10)))) * ( ((a2[0] * c2[0])/b2[0]) + ( (((CO2_ADULT_GAIN * currPersons * R)/(CO2_MASS * currVolume)) + ((CO2_OUTDOOR * PRESSURE_OUTDOOR * ACH)/(TEMP_OUTDOOR))) * (index/10)));
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

    cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
    
    float *a1, *b1, *c1, *d1, *a2, *b2, *c2, *d2; //host copies of a,b,c vectors
    float *d_a1, *d_b1, *d_c1, *d_d1, *d_a2, *d_b2, *d_c2, *d_d2; //device copies of a,b,c etc

    int size = N * sizeof(float); //size of memory that needs to be allocated

    //Allocate space for device copies of a,b,c, e
    cudaMalloc((void **)&d_a1, size);
    cudaMalloc((void **)&d_b1, size);
    cudaMalloc((void **)&d_c1, size);
    cudaMalloc((void **)&d_d1, size);
    cudaMalloc((void **)&d_a2, size);
    cudaMalloc((void **)&d_b2, size);
    cudaMalloc((void **)&d_c2, size);
    cudaMalloc((void **)&d_d2, size);

    //Allows device to get access to memory
    cudaHostAlloc( (void**)&a1, size, cudaHostAllocDefault);	 
    cudaHostAlloc( (void**)&b1, size, cudaHostAllocDefault);
    cudaHostAlloc( (void**)&c1, size, cudaHostAllocDefault);
    cudaHostAlloc( (void**)&d1, size, cudaHostAllocDefault);
    cudaHostAlloc( (void**)&a2, size, cudaHostAllocDefault);	 
    cudaHostAlloc( (void**)&b2, size, cudaHostAllocDefault);
    cudaHostAlloc( (void**)&c2, size, cudaHostAllocDefault);
    cudaHostAlloc( (void**)&d2, size, cudaHostAllocDefault);

    /*Loading CSV
        Room Structure:
            - A304: 1 - 41500
            - C114: 41501 - 83000 
    */
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

        if(N/2 < 41500) {
            a1[i] = float(gasDouble);
            b1[i] = float(tempDouble);
            c1[i] = float(pressureDouble);
        } else {
            a2[i] = float(gasDouble);
            b2[i] = float(tempDouble);
            c2[i] = float(pressureDouble);
        }

        i++;
				
    }
    
    printf("# OF DATA: %d \n", i - 1);
	ip.close();
    

    //Timing
    struct timeval t1, t2;
    gettimeofday(&t1, 0);

    for(int i=0;i < N;i++)
	{   // loop over data in chunks
	    // stream 1
		cudaMemcpyAsync(d_a1, a1, size, cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_b1, b1, size, cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_c1, c1, size, cudaMemcpyHostToDevice, stream1);
		getGasModelKernel1<<<1,N,0,stream1>>>(d_a1, d_b1, d_c1, d_d1);
		cudaMemcpyAsync(d1, d_d1, size, cudaMemcpyDeviceToHost, stream1);

		//stream 2
		cudaMemcpyAsync(d_a2, a2, size, cudaMemcpyHostToDevice, stream2);
		cudaMemcpyAsync(d_b2, b2, size, cudaMemcpyHostToDevice, stream2);
        cudaMemcpyAsync(d_c2, c2, size, cudaMemcpyHostToDevice, stream2);
        getGasModelKernel2<<<1,N,1,stream2>>>(d_a2, d_b2, d_c2, d_d2);
		cudaMemcpyAsync(d2, d_c2, size, cudaMemcpyDeviceToHost, stream2);
    }
    
	cudaStreamSynchronize(stream1); // wait for stream1 to finish
	cudaStreamSynchronize(stream2); // wait for stream2 to finish

    gettimeofday(&t2, 0);
    float time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    printf("EXECUTION TIME:  %5.4f ms \n", time);

    //A304
    //Some values of real CO2 from 3000 to 3010
    printf("MODEL CO2 - A304 = [");
    for (i=25000; i<25010; i++) printf(" %4.3f", d1[i]);
    printf(" ...]\n");

    //Some values of real CO2 from 3000 to 3010
    printf("REAL CO2 - A304= [");
    for (i=25000; i<25010; i++) printf(" %4.3f", a1[i]);
    printf(" ...]\n");
    
    
    //C114
    //Some values of real CO2 from 3000 to 3010
    printf("MODEL CO2 - C114 = [");
    for (i=25000; i<25010; i++) printf(" %4.3f", d2[i]);
    printf(" ...]\n");

    //Some values of real CO2 from 3000 to 3010
    printf("REAL CO2 - C114 = [");
    for (i=25000; i<25010; i++) printf(" %4.3f", a2[i]);
    printf(" ...]\n");


    //Destroying stream used
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
	return 0;
} 