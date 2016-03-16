#include "THCUNN.h"
#include "common.h"

#define MILTYPE_MAX 1
#define MILTYPE_NOR 2
#define MILTYPE_MAXNOR 3

__global__ void forward_kernel(const int nthreads, const float *input_data, float *output_data, 
                               const int batch_size, const int num_channels,
                               const int width, const int height, int mil_type) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int k, l;
        long offset;
        float prob, max_prob;
        
        offset = index * width * height;
        
        switch(mil_type) {
            case MILTYPE_MAX:
                prob = -FLT_MAX;
                for(k=0; k<height; k++){
                    for(l=0; l<width; l++){
                        prob = max(prob, input_data[offset]);
                        offset++;
                    }
                }
                output_data[index] = prob;
                break;
            
            case MILTYPE_NOR:
                prob = 1.;
                for(k=0; k<height; k++){
                    for(l=0; l<width; l++){
                        prob = prob*(1. - input_data[offset]);
                        offset++;
                    }
                }
                output_data[index] = 1. - prob;
                break;
            
            case MILTYPE_MAXNOR:
                prob = 1.;
                max_prob = -FLT_MAX;
                for(k=0; k<height; k++){
                    for(l=0; l<width; l++){
                        prob = prob*(1. - input_data[offset]);
                        max_prob = max(max_prob, input_data[offset]);
                        offset++;
                    }
                }
                output_data[index] = max(1. - prob, max_prob);
                break;
            
            default:
                break;
        }
    }
}
    
__global__ void backward_kernel(const int nthreads, const float *input_data, float *output_data, 
                                float *gradOutput_data, float *gradInput_data,
                                const int batch_size, const int num_channels,
                                const int width, const int height, int mil_type) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int k, l;
        long offset;
        float temp;
        
        offset = index * width * height;
        
        switch(mil_type) {
            case MILTYPE_MAX:
                for(k=0; k<height; k++){
                    for(l=0; l<width; l++){
                        gradInput_data[offset] = gradOutput_data[index] * (output_data[index] == input_data[offset]);
                        offset++;
                    }
                }
                break;
            
            case MILTYPE_NOR:
                for(k=0; k<height; k++){
                    for(l=0; l<width; l++){
                        temp = (1. - output_data[index])/(1. - input_data[offset]);
                        gradInput_data[offset] = gradOutput_data[index] * temp;
                        offset++;
                    }
                }
                break;
            
            case MILTYPE_MAXNOR:
                for(k=0; k<height; k++){
                    for(l=0; l<width; l++){
                        temp = min(1., (1. - output_data[index])/(1. - input_data[offset]));
                        gradInput_data[offset] = gradOutput_data[index] * temp;
                        offset++;
                    }
                }
                break;
            
            default:
                break;
        }
    }
}
    
void THNN_CudaSpatialMIL_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output, int mil_type)
{
    THCUNN_assertSameGPU(state, 2, input, output);
    
    long batch_size;
    long num_channels;
    long width;
    long height;
    float *input_data, *output_data;
    int count;
    
    batch_size = input->size[0];
    num_channels = input->size[1];
    height = input->size[2];
    width = input->size[3];
    
    input = THCudaTensor_newContiguous(state, input);
    
    THCudaTensor_zero(state, output);
    THCudaTensor_resize2d(state, output, batch_size, num_channels);
    THArgCheck(THCudaTensor_isContiguous(state, output), 2, "Output must be contiguous");
    
    input_data = THCudaTensor_data(state, input);
    output_data = THCudaTensor_data(state, output);
    
    count = THCudaTensor_nElement(state, output);
    
    forward_kernel<<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>> 
                (count, input_data, output_data,
                 batch_size, num_channels, width, height, mil_type);
    
    THCudaTensor_free(state, input);
}

void THNN_CudaSpatialMIL_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *output, THCudaTensor *gradOutput, THCudaTensor *gradInput, int mil_type)
{
    long batch_size;
    long num_channels;
    long width;
    long height;
    float *input_data, *output_data, *gradOutput_data, *gradInput_data;
    int count;
    
    batch_size = input->size[0];
    num_channels = input->size[1];
    height = input->size[2];
    width = input->size[3];
    
    input = THCudaTensor_newContiguous(state, input);
    output = THCudaTensor_newContiguous(state, output);
    gradOutput = THCudaTensor_newContiguous(state, gradOutput);
    
    THCudaTensor_resizeAs(state, gradInput, input);
    THArgCheck(THCudaTensor_isContiguous(state, gradInput), 2, "Output must be contiguous");
    
    input_data = THCudaTensor_data(state, input);
    output_data = THCudaTensor_data(state, output);
    gradOutput_data = THCudaTensor_data(state, gradOutput);
    gradInput_data = THCudaTensor_data(state, gradInput);
    
    THCudaTensor_zero(state, gradInput);
    
    count = THCudaTensor_nElement(state, output);
    
    backward_kernel<<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>> 
                (count, input_data, output_data, gradOutput_data, gradInput_data,
                 batch_size, num_channels, width, height, mil_type);
    
    THCudaTensor_free(state, input);
    THCudaTensor_free(state, output);
    THCudaTensor_free(state, gradOutput);
}


