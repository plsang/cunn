#include "THCUNN.h"
#include "common.h"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>


__global__ void forward_kernel(const int nthreads, const float *input_data, const float *target_data, float *sumarr_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        
        float eps = 1e-5;
    
        CUDA_KERNEL_LOOP(index, nthreads) {

            if(target_data[index] == 1){
                sumarr_data[index] = -log(max(input_data[index], eps));
            }
            else{
                sumarr_data[index] = -log(max(1-input_data[index], eps));
            }

        }
        
    }
}
    
__global__ void backward_kernel(const int nthreads, const float *input_data, const float *target_data, 
                                float *gradInput_data, const float loss_weight) {
                                
   CUDA_KERNEL_LOOP(index, nthreads) {
        
        float eps = 1e-5;
    
        CUDA_KERNEL_LOOP(index, nthreads) {
            float grad;

            if(target_data[index] == 1){
                grad = -loss_weight/(max(input_data[index], eps) * nthreads);
            }
            else{
                grad = loss_weight/(max(1-input_data[index], eps) * nthreads);
            }
            gradInput_data[index] = grad;
        }
        
    }
}
    
void THNN_CudaMultiLabelCrossEntropyCriterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *output)
{
    long num_element;
    float loss;
    float *input_data, *target_data, *sumarr_data;
    THCudaTensor *sumarr;
    
    input = THCudaTensor_newContiguous(state, input);
    target = THCudaTensor_newContiguous(state, target);
    
    sumarr = THCudaTensor_newClone(state, input);
    sumarr = THCudaTensor_newContiguous(state, sumarr);
    THCudaTensor_zero(state, output);
    
    input_data = THCudaTensor_data(state, input);
    target_data = THCudaTensor_data(state, target);
    sumarr_data = THCudaTensor_data(state, sumarr);

    num_element = THCudaTensor_nElement(state, input);
    
    forward_kernel<<< GET_BLOCKS(num_element), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>> 
                (num_element, input_data, target_data, sumarr_data);
    
    // wrap raw pointer with a device_ptr 
    thrust::device_ptr<float> sumarr_ptr(sumarr_data);
    // summing
    loss = thrust::reduce(sumarr_ptr, sumarr_ptr+num_element, (float) 0);
    
    loss = loss/num_element;
    
    THCudaTensor_set1d(state, output, 0, loss);
    
    THCudaTensor_free(state, input);
    THCudaTensor_free(state, target);
    THCudaTensor_free(state, sumarr);
}

void THNN_CudaMultiLabelCrossEntropyCriterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *gradInput,
          float loss_weight)
{
    long num_element;
    float *input_data, *target_data, *gradInput_data;
    
    input = THCudaTensor_newContiguous(state, input);
    target = THCudaTensor_newContiguous(state, target);
    
    THCudaTensor_resizeAs(state, gradInput, input);
    THCudaTensor_zero(state, gradInput);
    
    input_data = THCudaTensor_data(state, input);
    target_data = THCudaTensor_data(state, target);
    gradInput_data = THCudaTensor_data(state, gradInput);

    num_element = THCudaTensor_nElement(state, input);
    
    backward_kernel<<< GET_BLOCKS(num_element), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>> 
                (num_element, input_data, target_data, gradInput_data, loss_weight);
   
    THCudaTensor_free(state, input);
    THCudaTensor_free(state, target);
}

