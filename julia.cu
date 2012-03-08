////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
// julia.cu (c) Stephen Smithbower 2012
//
// CPU and Cuda implementations of the Julia set associated with the Newton 
// iteration for the complex function f(z) = z^3 - 1.
//
// Utilizes [libbmp - BMP library] for image saving. The main source for this
// can be found @ http://code.google.com/p/libbmp/
////////////////////////////////////////////////////////////////////////////////
// -w    Image width, default 1024.
// -h    Image height, default 768.
// -z    Zoom level, default 2.2
// -i    Maximum number of Newton's method iterations, default 200.
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//*************
// This code is presented as-is, without warrenty or support. You are welcome
// to use this code in whatever way you wish, however I would appreciate
// some credit =)
//*************

////////////////////////////////////////////////////////////////////////////////
// Includes
////////////////////////////////////////////////////////////////////////////////
#include <math.h>
#include <cuda_runtime.h>

#include "bmpfile.h"
#include "bmpfile.c"


////////////////////////////////////////////////////////////////////////////////
// Default Globals
////////////////////////////////////////////////////////////////////////////////
int width = 1024;           //Width of the image, in pixels.
int height = 768;           //Height of the image, in pixels.

int max_iterations = 200;    //Maximum number iterations of Newton's method.
                            //200 seems to work out pretty well.

float zoom = 2.2f;          //Viewing region (-x : x).

float epsilon = 0.01f;     //Maximum difference when comparing floats.


////////////////////////////////////////////////////////////////////////////////
// Function Prototypes
////////////////////////////////////////////////////////////////////////////////
void cpu_julia(int *matrix);
__global__ void gpu_julia(int *matrix, int width, int height, int max_iterations, float zoom, float epsilon); 


////////////////////////////////////////////////////////////////////////////////
// Complex Number Helper Functions
////////////////////////////////////////////////////////////////////////////////
__device__ __host__ void complex_add(float a, float b, float c, float d, float *realOut, float *imgOut)
{
    *realOut = a + c;
    *imgOut = b + d;
}

__device__ __host__ void complex_sub(float a, float b, float c, float d, float *realOut, float *imgOut)
{
    *realOut = a - c;
    *imgOut = b - d;
}

__device__ __host__ void complex_mul(float a, float b, float c, float d, float *realOut, float *imgOut)
{
    *realOut = (a * c) - (b * d);
    *imgOut = (b * c) + (a * d);
}

__device__ __host__ void complex_div(float a, float b, float c, float d, float *realOut, float *imgOut)
{
    *realOut = ((a * c) + (b * d)) / (pow(c, 2) + pow(d, 2));
    *imgOut = ((b * c) - (a * d))/ (pow(c, 2) + pow(d, 2));
}


////////////////////////////////////////////////////////////////////////////////
// Bitmap Helper Function
////////////////////////////////////////////////////////////////////////////////
void write_bitmap(int *matrix, char *filename)
{
    bmpfile_t *bmp;
    bmp = bmp_create(width, height, 24);
    rgb_pixel_t pixel;

    for (int x = 0; x < width; x++)
    for (int y = 0; y < height; y++)
    {
        pixel.red = 0;
        pixel.green = 0;
        pixel.blue = 0;

        switch(matrix[x * height + y])
        {
            case 1:
                pixel.red = 255;
                break;

            case 2:
                pixel.blue = 255;
                break;

            case 3:
                pixel.green = 255;
                break;
        }

        bmp_set_pixel(bmp, x, y, pixel);
    }

    bmp_save(bmp, filename);
    bmp_destroy(bmp);
}


////////////////////////////////////////////////////////////////////////////////
// Main Function
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) 
{
    //Input parameters.
    for (int i = 0; i < argc; i++)
    {
        if (argv[i][0] == '-')
        {
            if (argv[i][1] == 'w') //Width of the image.
                width = atoi(argv[++i]);
            if (argv[i][1] == 'h') //Height of the image.
                height = atoi(argv[++i]);
            if (argv[i][1] == 'z') //Zoom level.
                zoom = atof(argv[++i]);
            if (argv[i][1] == 'i') //Maximum number of iterations.
                max_iterations = atoi(argv[++i]);
            if (argv[i][1] == 'e') //Set epsilon - controls calculation precision.
                epsilon = atof(argv[++i]);
        }
    }

    int dif_count = 0;

    ///////////
    // Initialize memory.
    ///////////
    //Host.
    int *cpu_image = (int*)malloc(sizeof(int) * width * height);
    int *gpu_image = (int*)malloc(sizeof(int) * width * height);

    //Device.
    int *device_image;
    cudaMalloc((void**)&device_image, sizeof(int) * width * height);

    ///////////
    // Perform CPU calculations (gold).
    ///////////
    cpu_julia(cpu_image);
    write_bitmap(cpu_image, "cpu.bmp");

    ///////////
    // Perform GPU calculations, 128 threads per block (arbitrary).
    ///////////
    gpu_julia<<<ceil((float)width * (float)height / 128.0f), 128>>>(device_image, width, height, max_iterations, zoom, epsilon);
    cudaThreadSynchronize();

    cudaMemcpy(gpu_image, device_image, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
    write_bitmap(gpu_image, "gpu.bmp");

    ///////////
    // Validate GPU results.
    ///////////
    for (int x = 0; x < width; x++)
    for (int y = 0; y < height; y++)
    {
        int index = (x * height) + y;

        if (cpu_image[index] != gpu_image[index])
            dif_count++;
    }

    if (dif_count < (width * height * 0.01f)) //Fewer than 1% difference.
        printf("GPU Passes!\n");
    else
        printf("GPU FAILS =(\n");

    ///////////
    // Memory cleanup.
    ///////////
    free(cpu_image);
    free(gpu_image);

    cudaFree(device_image);

    return 0;
}


////////////////////////////////////////////////////////////////////////////////
// CPU Implementation
////////////////////////////////////////////////////////////////////////////////
void cpu_julia(int *matrix)
{
    float newRe, newIm, oldRe, oldIm;
    float z_3_r, z_3_i, z_2_r, z_2_i, inner_r, inner_i;

    float ratio = (float)height / (float)width;

    for(int x = 0; x < width; x++)
    for(int y = 0; y < height; y++)
    {
        ///////////
        // Set up starting value based on x, y (x = real, y = imaginary).
        ///////////
        newRe = (((float)x / (float)width) - 0.5f) * 2.0f * zoom;
        newIm = ratio * (((float)y / (float)height) - 0.5f) * 2.0f * zoom;

        ///////////
        // Newton's Method. z[+ 1] = z - ((z^3 - 1) / 3z^2)
        ///////////
        for(int i = 0; i < max_iterations; i++)
        {
            oldRe = newRe;
            oldIm = newIm;

            //Clear everything.
            z_3_r = z_3_i = z_2_r = z_2_i = inner_r = inner_i = 0;

            complex_mul(oldRe, oldIm, oldRe, oldIm, &z_2_r, &z_2_i); // z^2
            complex_mul(z_2_r, z_2_i, oldRe, oldIm, &z_3_r, &z_3_i); // z^3
            z_3_r -= 1.0f; //z^3 - 1

            z_2_r *= 3.0f; // 3z^2
            z_2_i *= 3.0f;

            complex_div(z_3_r, z_3_i, z_2_r, z_2_i, &inner_r, &inner_i); // ((z^3 - 1) / 3z^2)

            complex_sub(oldRe, oldIm, inner_r, inner_i, &newRe, &newIm); //z - ((z^3 - 1) / 3z^2)

            //If we've mostly converged, break out early.
            if (abs(newRe - oldRe) < epsilon && abs(newIm - oldIm) < epsilon)
                break;
        }

        ///////////
        // Figure out which root we've converged to.
        ///////////
        if (abs(1.0f - newRe) < epsilon && abs(0 - newIm) < epsilon)
            matrix[x * height + y] = 1;
        else
            if (newRe - 0.5f < epsilon && 0.86603f -  newIm < epsilon)
                matrix[x * height + y] = 2;
            else
                if (newRe - 0.5f < epsilon && newIm - 0.86603f < epsilon)
                    matrix[x * height + y] = 3;
                else
                    matrix[x * height + y] = 0;
    }
}


////////////////////////////////////////////////////////////////////////////////
// GPU Implementation
////////////////////////////////////////////////////////////////////////////////
__global__ void gpu_julia(int *matrix, int width, int height, int max_iterations, float zoom, float epsilon)
{
    //Compute global thread id to index global memory.
    //Each thread is one pixel.
    int threadID = (blockIdx.x * blockDim.x) + threadIdx.x;

    float newRe, newIm, oldRe, oldIm;
    float z_3_r, z_3_i, z_2_r, z_2_i, inner_r, inner_i;

    //Guard to make sure we're not writing to memory we don't own.
    if (threadID < width * height)
    {
        ///////////
        // Set up starting value based on x, y (x = real, y = imaginary).
        ///////////
        int x = (threadID / height);
        int y = (threadID % height);

        newRe = (((float)x / (float)width) - 0.5f) * 2.0f * zoom;
        newIm = ((float)height / (float)width) * (((float)y / (float)height) - 0.5f) * 2.0f * zoom;

        ///////////
        // Newton's Method. z[+ 1] = z - ((z^3 - 1) / 3z^2)
        ///////////
        for(int i = 0; i < max_iterations; i++)
        {
            //Clear everything.
            z_3_r = z_3_i = z_2_r = z_2_i = inner_r = inner_i = 0;

            oldRe = newRe;
            oldIm = newIm;

            complex_mul(oldRe, oldIm, oldRe, oldIm, &z_2_r, &z_2_i); // z^2
            complex_mul(z_2_r, z_2_i, oldRe, oldIm, &z_3_r, &z_3_i); // z^3
            z_3_r -= 1.0f; //z^3 - 1

            z_2_r *= 3.0f; // 3z^2
            z_2_i *= 3.0f;

            complex_div(z_3_r, z_3_i, z_2_r, z_2_i, &inner_r, &inner_i); // ((z^3 - 1) / 3z^2)

            complex_sub(oldRe, oldIm, inner_r, inner_i, &newRe, &newIm); //z - ((z^3 - 1) / 3z^2)

            //If we've mostly converged, break out early.
            if (abs(newRe - oldRe) < epsilon && abs(newIm - oldIm) < epsilon)
                break;
        }

        ///////////
        // Figure out which root we've converged to.
        ///////////
        if (abs(1.0f - newRe) < epsilon && abs(0 - newIm) < epsilon)
            matrix[threadID] = 1;
        else
            if (newRe - 0.5f < epsilon && 0.86603f -  newIm < epsilon)
                matrix[threadID] = 2;
            else
                if (newRe - 0.5f < epsilon && newIm - 0.86603f < epsilon)
                    matrix[threadID] = 3;
                else
                    matrix[threadID] = 0;
    }
}