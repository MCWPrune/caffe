/***************************** MulticoreWare_Modified - Feature: Pruning / Splicing ************************************/
#include <vector>
#include <cmath>

#include "caffe/filler.hpp"
#include "caffe/layers/squeeze_conv_layer.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
void SqueezeConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer <Dtype>::LayerSetUp(bottom, top);

  SqueezeConvolutionParameter sqconv_param = this->layer_param_.squeeze_convolution_param();

  if(this->blobs_.size()==2 && (this->bias_term_)){
    this->blobs_.resize(4);

    // Intialize and fill the weightmask & biasmask
    this->blobs_[2].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
    shared_ptr<Filler<Dtype> > weight_mask_filler(GetFiller<Dtype>(
        sqconv_param.weight_mask_filler()));
    weight_mask_filler->Fill(this->blobs_[2].get());
    this->blobs_[3].reset(new Blob<Dtype>(this->blobs_[1]->shape()));
    shared_ptr<Filler<Dtype> > bias_mask_filler(GetFiller<Dtype>(
        sqconv_param.bias_mask_filler()));
    bias_mask_filler->Fill(this->blobs_[3].get());
  }
  else if(this->blobs_.size()==1 && (!this->bias_term_)){
    this->blobs_.resize(2);
    // Intialize and fill the weightmask
    this->blobs_[1].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
    shared_ptr<Filler<Dtype> > weight_mask_filler(GetFiller<Dtype>(
        sqconv_param.weight_mask_filler()));
    weight_mask_filler->Fill(this->blobs_[1].get());
  }

  // Intializing the tmp tensor
  this->weight_tmp_.Reshape(this->blobs_[0]->shape());
  if(this->bias_term_)
    this->bias_tmp_.Reshape(this->blobs_[1]->shape());

  // Intialize the hyper-parameters
  this->std = 0;this->mu = 0;
  this->gamma = sqconv_param.gamma();
  this->power = sqconv_param.power();
  this->crate = sqconv_param.c_rate();
  this->iter_stop_ = sqconv_param.iter_stop();
}

template <typename Dtype>
void SqueezeConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void SqueezeConvolutionLayer<Dtype>::AggregateParams(const int n, const Dtype* wb, const Dtype* mask,
    Dtype* mu, Dtype* std, unsigned int *count) {
  for (unsigned int k = 0;k < n; ++k) {
    this->mu  += fabs(mask[k] * wb[k]);
    this->std += mask[k] * wb[k] * wb[k];
    if (mask[k] * wb[k] != 0) (*count)++;
  }
}

template <typename Dtype>
void SqueezeConvolutionLayer<Dtype>::CalculateMask(const int n, const Dtype* wb, Dtype* mask,
    Dtype mu, Dtype std, Dtype r) {
  for (unsigned int k = 0;k < n;++k) {
        // The constants 0.9 and 1.1 is to set margin that witholds few parameters undergoing pruning / splicing
        if (mask[k] == 1 && fabs(wb[k]) <= 0.9 * r *  std::max(mu + std, Dtype(0)))
          mask[k] = 0; //Pruning
        else if (mask[k] == 0 && fabs(wb[k]) > 1.1 * r * std::max(mu + std, Dtype(0)))
          mask[k] = 1; //Splicing
      }
}

template <typename Dtype>
void SqueezeConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = NULL;
  Dtype* weightMask = NULL;
  Dtype* weightTmp = NULL;
  const Dtype* bias = NULL;
  Dtype* biasMask = NULL;
  Dtype* biasTmp = NULL;
  int maskcount = 0;
  if (this->bias_term_) {
    weight = this->blobs_[0]->mutable_cpu_data();
    weightMask = this->blobs_[2]->mutable_cpu_data();
    weightTmp = this->weight_tmp_.mutable_cpu_data();
    bias = this->blobs_[1]->mutable_cpu_data();
    biasMask = this->blobs_[3]->mutable_cpu_data();
    biasTmp = this->bias_tmp_.mutable_cpu_data();
    maskcount = this->blobs_[2]->count();
  }
  else {
    weight = this->blobs_[0]->mutable_cpu_data();
    weightMask = this->blobs_[1]->mutable_cpu_data();
    weightTmp = this->weight_tmp_.mutable_cpu_data();
    maskcount = this->blobs_[1]->count();
  }
  vector<int> index_zero;

  // Core logic for Pruning/Splicing
  if (this->phase_ == TRAIN) {
    //To avoid corrupted mask value
    for (int l =0; l<maskcount; l++)
    {
      if (weightMask[l] !=0 && weightMask[l]!= 1)
      {
        weightMask[l] = abs(round(weightMask[l]));
      }
    }
    // Calculate the mean and standard deviation of learnable parameters 
    if ((this->std == 0 && this->iter_ == 0) || this->iter_== 40 || this->iter_== 80 || this->iter_== 120 || this->iter_== 160) {
      unsigned int ncount = 0;
      AggregateParams(this->blobs_[0]->count(), weight, weightMask, &mu, &std, &ncount);
      if (this->bias_term_) {
        AggregateParams(this->blobs_[1]->count(), bias, biasMask, &mu, &std, &ncount);
      }
      this->mu /= ncount; this->std -= ncount * mu * mu;
      this->std /= ncount; this->std = sqrt(std);
    }

// No pruning/splicing during Retraining
#if !RETRAINING 
  // Calculate the weight mask and bias mask with probability
    Dtype r = static_cast<Dtype>(rand())/static_cast<Dtype>(RAND_MAX);
    if (pow(1 + (this->gamma) * (this->iter_), -(this->power)) > r && (this->iter_) < (this->iter_stop_)) {
      CalculateMask(this->blobs_[0]->count(), weight, weightMask, this->mu, this->std, this->crate);  
      if (this->bias_term_) {       
        CalculateMask(this->blobs_[1]->count(), bias, biasMask, this->mu, this->std, this->crate);
      } 
    }
  #endif

// Dynamic Splicing
// Randomly unprune the pruned weights based on the splicing ratio
#if DYNAMIC_SPLICING
    if (this->iter_ == 0) {
      vector<int> index_zero;

      for (unsigned int k = 0; k < this->blobs_[0]->count(); ++k) {
        if(weightMask[k] == 0)
          index_zero.push_back(k);
      }
      int zero_count = index_zero.size();
      int to_bespliced = zero_count * CONV_SPLICING_RATE;
      std::random_shuffle(index_zero.begin(), index_zero.end());

      for (unsigned int k = 0; k < to_bespliced; ++k)
        weightMask[index_zero[k]] = 1;
    }
#endif
  }

  // Calculate the current (masked) weight and bias
  for (unsigned int k = 0;k < this->blobs_[0]->count(); ++k) {
    weightTmp[k] = weight[k]*weightMask[k];
  }
  if (this->bias_term_){
    for (unsigned int k = 0;k < this->blobs_[1]->count(); ++k) {
      biasTmp[k] = bias[k]*biasMask[k];
    }
  }

  // Forward calculation with (masked) weight and bias 
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + bottom[i]->offset(n), weightTmp,
          top_data + top[i]->offset(n));
      if (this->bias_term_) {
        this->forward_cpu_bias(top_data + top[i]->offset(n), biasTmp);
      }
    }
  }
}

template <typename Dtype>
void SqueezeConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weightTmp = this->weight_tmp_.cpu_data();  
  const Dtype* weightMask = NULL;
  if(this->bias_term_)
    weightMask = this->blobs_[2]->cpu_data();
  else
    weightMask = this->blobs_[1]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      const Dtype* biasMask = this->blobs_[3]->cpu_data();
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (unsigned int k = 0;k < this->blobs_[1]->count(); ++k) {
        bias_diff[k] = bias_diff[k]*biasMask[k];
      }
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      for (unsigned int k = 0;k < this->blobs_[0]->count(); ++k) {
        weight_diff[k] = weight_diff[k]*weightMask[k];
      }
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + top[i]->offset(n), weightTmp,
              bottom_diff + bottom[i]->offset(n));
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SqueezeConvolutionLayer);
#endif

INSTANTIATE_CLASS(SqueezeConvolutionLayer);
//REGISTER_LAYER_CLASS(SqueezeConvolution);

}  // namespace caffe
/***********************************************************************************************************************/
