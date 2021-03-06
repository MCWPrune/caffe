/***************************** MulticoreWare_Modified - Feature: Pruning / Splicing ************************************/
#include <vector>
#include <cmath>
#include<stdio.h>

#include "caffe/filler.hpp"
#include "caffe/layers/squeeze_conv_layer.hpp"


namespace caffe {

// The constant NUM_THREADS should be equal to the value in SqueezeCMomentCalc
template <typename Dtype>
__global__ void SqueezeCMomentCollect(const int n, const Dtype* wb, const Dtype* mask,
    Dtype* mu, Dtype* std, unsigned int* count ) {
  const int NUM_THREADS = 512;
  __shared__ Dtype param [4 * NUM_THREADS];
  __shared__ unsigned int tcount [2 * NUM_THREADS];
  unsigned int t = threadIdx.x; 
  unsigned int s = 2 * blockIdx.x * NUM_THREADS;
  if (s + t < n){
    param[t] = fabs(mask[s + t] * wb[s + t]);
    param[t + 2 * NUM_THREADS] = mask[s + t] * wb[s + t] * wb[s + t];
    if(mask[s + t] * wb[s + t] != 0) tcount[t] = 1;
    else tcount[t] = 0;
  }
  else{
    param[t] = 0;param[t + 2 * NUM_THREADS] = 0;tcount[t] = 0;
  }
  if (s + t + NUM_THREADS < n){
    param[t + NUM_THREADS] = fabs(mask[s + t + NUM_THREADS] * wb[s + t + NUM_THREADS]);
    param[t + 3 * NUM_THREADS] = mask[s + t + NUM_THREADS] * wb[s + t + NUM_THREADS] * wb[s + t + NUM_THREADS];
    if(mask[s + t +NUM_THREADS] * wb[s + t + NUM_THREADS] != 0) tcount[t + NUM_THREADS] = 1;
    else tcount[ t + NUM_THREADS] = 0;
  }
  else{
    param[t + NUM_THREADS] = 0; param[t + 3 * NUM_THREADS] = 0; tcount[t + NUM_THREADS] = 0;
  }
  __syncthreads(); 
  for(unsigned int stride = NUM_THREADS; stride >= 1; stride >>= 1) {
    if (t < stride ){
      param[t] += param[t + stride]; 
      param[t + 2 * NUM_THREADS] += param[t + 2 * NUM_THREADS + stride];
      tcount[t] += tcount[t + stride];
    }
    __syncthreads();
  }
  if (t == 0){
    mu   [blockIdx.x] = param[0];
    std  [blockIdx.x] = param[2 * NUM_THREADS];
    count[blockIdx.x] = tcount[0]; 
  }
}

// The constant NUM_THREADS should be equal to the value in SqueezeCMomentCalc
template <typename Dtype>
__global__ void SqueezeCNzeroCollect(const int n, const Dtype* mask, unsigned int* count ) {
  const int NUM_THREADS = 512;  
  __shared__ unsigned int tcount [2 * NUM_THREADS];
  unsigned int t = threadIdx.x; 
  unsigned int s = 2 * blockIdx.x * NUM_THREADS;
  tcount[t] = 0;
  if (s + t < n && mask[s + t] != 0){
    tcount[t] = 1;
  }
  tcount[t+NUM_THREADS] = 0;
  if (s + t + NUM_THREADS < n && mask[s + t + NUM_THREADS] != 0){
    tcount[t + NUM_THREADS] = 1;
  }
  __syncthreads(); 
  for(unsigned int stride = NUM_THREADS; stride >= 1; stride >>= 1) {
    if (t < stride ){
      tcount[t] += tcount[t + stride];
    }
    __syncthreads();
  }
  if (t == 0){
    count[blockIdx.x] = tcount[0];
  }
}

//Check condition for pruning and splicing
template <typename Dtype>
__global__ void SqueezeCMaskCalc(const int n, const Dtype* wb,
    Dtype* mask, Dtype mu, Dtype std, Dtype r) {
  CUDA_KERNEL_LOOP(index, n) {
    // The constants 0.9 and 1.1 is to set margin that witholds few parameters undergoing pruning / splicing
    if (mask[index] == 1 && fabs(wb[index]) <= 0.9 * r * max(mu + std, Dtype(0))) {
      mask[index] = 0;
    }
    else if (mask[index] == 0 && fabs(wb[index]) > 1.1 * r * max(mu + std, Dtype(0))){
      mask[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void SqueezeCMaskApply(const int n, const Dtype* wb,
    const Dtype* mask, Dtype* wb_t) {
  CUDA_KERNEL_LOOP(index, n) {
    wb_t[index] = wb[index] * mask[index];
  }
}

template <typename Dtype>
__global__ void ValidateMask(const int n,  Dtype* wb) {
  CUDA_KERNEL_LOOP(index, n) {
  if (wb[index] !=0 && wb[index]!= 1)
    wb[index] = fabs(rintf(wb[index]));
  }
}

//Calculate Mean and std deviation of weights 
template <typename Dtype>
void SqueezeCMomentCalc(const int n, const Dtype* wb, const Dtype* mask, Dtype* mu, Dtype* std, unsigned int* ncount){
  const unsigned int NUM_THREADS = 512;
  Dtype* pmu_g; Dtype* pstd_g; unsigned int* pncount_g;
  Dtype* pmu_c; Dtype* pstd_c; unsigned int* pncount_c;
  int num_p = (n + (NUM_THREADS << 1) - 1) / (NUM_THREADS << 1);
  cudaMalloc(&pmu_g, sizeof(Dtype)  * num_p);
  cudaMalloc(&pstd_g, sizeof(Dtype) * num_p);
  cudaMalloc(&pncount_g, sizeof(unsigned int) * num_p);
  pmu_c = (Dtype*) malloc(num_p * sizeof(Dtype));
  pstd_c = (Dtype*) malloc(num_p * sizeof(Dtype));
  pncount_c = (unsigned int*) malloc(num_p * sizeof(unsigned int));
  SqueezeCMomentCollect<Dtype><<<num_p,NUM_THREADS>>>(n, wb, mask, pmu_g, pstd_g, pncount_g);
  CUDA_POST_KERNEL_CHECK;
  cudaMemcpy(pmu_c, pmu_g, sizeof(Dtype) * num_p, cudaMemcpyDeviceToHost);
  cudaMemcpy(pstd_c, pstd_g, sizeof(Dtype) * num_p, cudaMemcpyDeviceToHost);
  cudaMemcpy(pncount_c, pncount_g, sizeof(unsigned int) * num_p, cudaMemcpyDeviceToHost);
  for (int i = 0; i < num_p; i++) {
    *mu += pmu_c[i]; *std += pstd_c[i]; *ncount += pncount_c[i];
  }
  cudaFree(pmu_g);cudaFree(pstd_g);cudaFree(pncount_g);
  free(pmu_c);free(pstd_c);free(pncount_c);
}

//Count the number of non-zero weights
template <typename Dtype>
void SqueezeCNZeroCalc(const int n, const Dtype* mask, unsigned int* ncount ){
  const unsigned int NUM_THREADS = 512;
  unsigned int* pncount_g;
  unsigned int* pncount_c;
  int num_p = (n + (NUM_THREADS << 1) - 1) / (NUM_THREADS << 1);
  cudaMalloc(&pncount_g, sizeof(unsigned int) * num_p);
  pncount_c = (unsigned int*) malloc(num_p * sizeof(unsigned int));
  SqueezeCNzeroCollect<Dtype><<<num_p,NUM_THREADS>>>(n, mask, pncount_g);
  CUDA_POST_KERNEL_CHECK;
  cudaMemcpy(pncount_c, pncount_g, sizeof(unsigned int) * num_p, cudaMemcpyDeviceToHost);
  for (int i = 0; i < num_p; i++) {
    *ncount += pncount_c[i];
  }
  cudaFree(pncount_g);
  free(pncount_c);
}

template <typename Dtype>
void SqueezeConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = NULL;
  Dtype* weightMask = NULL;
  Dtype* weightTmp = NULL;
  const Dtype* bias = NULL;
  Dtype* biasMask = NULL;
  Dtype* biasTmp = NULL;
  int maskcount = 0;
  if (this->bias_term_) {
    weight = this->blobs_[0]->mutable_gpu_data();
    weightMask = this->blobs_[2]->mutable_gpu_data();
    weightTmp = this->weight_tmp_.mutable_gpu_data();
    bias = this->blobs_[1]->mutable_gpu_data();
    biasMask = this->blobs_[3]->mutable_gpu_data();
    biasTmp = this->bias_tmp_.mutable_gpu_data();
    maskcount = this->blobs_[2]->count();
  }
  else {
    weight = this->blobs_[0]->mutable_gpu_data();
    weightMask = this->blobs_[1]->mutable_gpu_data();
    weightTmp = this->weight_tmp_.mutable_gpu_data();
    maskcount = this->blobs_[1]->count();
  }
  
  if (this->phase_ == TRAIN) {
  
      // Validate mask value to avoid corrupted mask value
    ValidateMask<Dtype><<<CAFFE_GET_BLOCKS(maskcount),
    CAFFE_CUDA_NUM_THREADS>>>(maskcount,weightMask);
    CUDA_POST_KERNEL_CHECK;

    // Calculate the mean and standard deviation of learnable parameters
    if ((this->std == 0 && this->iter_ == 0) || this->iter_== 40 || this->iter_== 80 || this->iter_== 120 || this->iter_== 160) {
      unsigned int ncount = 0;
      SqueezeCMomentCalc(this->blobs_[0]->count(), weight, weightMask, &mu, &std, &ncount);
      if (this->bias_term_) {
        SqueezeCMomentCalc(this->blobs_[1]->count(), bias, biasMask, &mu, &std, &ncount); 
      }
      this->mu /= ncount; this->std -= ncount * mu * mu; 
      this->std /= ncount; this->std = sqrt(std);
      LOG(INFO)<<mu<<"  "<<std<<"  "<<ncount<<"\n";
    }
// No pruning/splicing during Retraining
#if !RETRAINING 
    // Calculate the weight mask and bias mask with probability
    Dtype r = static_cast<Dtype>(rand())/static_cast<Dtype>(RAND_MAX);
    if (pow(1 + (this->gamma) * (this->iter_), -(this->power)) > r && (this->iter_) < (this->iter_stop_)) {
      SqueezeCMaskCalc<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()),
        CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[0]->count(), weight,
        weightMask, this->mu, this->std, this->crate);

      CUDA_POST_KERNEL_CHECK;
      if (this->bias_term_) {
        SqueezeCMaskCalc<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[1]->count()),
          CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[1]->count(), bias, 
          biasMask, this->mu, this->std, this->crate);
        CUDA_POST_KERNEL_CHECK;
      }
    }
#endif
// Dynamic Splicing
// Randomly unprune the pruned weights based on the splicing ratio
#if DYNAMIC_SPLICING
    if (this->iter_ == 0) {
      vector<int> index_zero;
      Dtype* weightMask_cpu = (Dtype *)malloc(this->blobs_[0]->count() *(sizeof(Dtype)));
      cudaMemcpy(weightMask_cpu, weightMask, this->blobs_[0]->count() *(sizeof(Dtype)), cudaMemcpyDeviceToHost);
      for (unsigned int k = 0; k < this->blobs_[0]->count(); ++k) {
          if(weightMask_cpu[k] == 0) {
              index_zero.push_back(k);
          }
      }
      int zero_count = index_zero.size();
      int to_bespliced = zero_count * CONV_SPLICING_RATE;
      std::random_shuffle(index_zero.begin(), index_zero.end());

      for (unsigned int k = 0; k < to_bespliced; ++k) {
          weightMask_cpu[index_zero[k]] = 1;
      }
    cudaMemcpy(weightMask, weightMask_cpu, this->blobs_[0]->count() *(sizeof(Dtype)), cudaMemcpyHostToDevice);
    free(weightMask_cpu);
    }
#endif
  }
 
  // Calculate the current (masked) weight and bias
  SqueezeCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()),
    CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[0]->count(), weight, weightMask, weightTmp);
  CUDA_POST_KERNEL_CHECK;
  if (this->bias_term_) {
    SqueezeCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[1]->count()),
      CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[1]->count(), bias, biasMask, biasTmp);
    CUDA_POST_KERNEL_CHECK;
  }

	// Forward calculation with (masked) weight and bias 
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + bottom[i]->offset(n), weightTmp,
          top_data + top[i]->offset(n));
      if (this->bias_term_) {
        this->forward_gpu_bias(top_data + top[i]->offset(n), biasTmp);
      }
    }
  }
}

template <typename Dtype>
void SqueezeConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weightTmp = this->weight_tmp_.gpu_data();
  const Dtype* weightMask = NULL;
  if(this->bias_term_)
    weightMask = this->blobs_[2]->gpu_data();
  else
    weightMask = this->blobs_[1]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      const Dtype* biasMask = this->blobs_[3]->gpu_data();
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      SqueezeCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[3]->count()),
        CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[3]->count(), bias_diff, biasMask, bias_diff);
      CUDA_POST_KERNEL_CHECK;
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      SqueezeCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()),
        CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[0]->count(), weight_diff, weightMask, weight_diff);
      CUDA_POST_KERNEL_CHECK; 			
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + top[i]->offset(n), weightTmp,
              bottom_diff + bottom[i]->offset(n));
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SqueezeConvolutionLayer);
}  // namespace caffe
/***********************************************************************************************************************/
