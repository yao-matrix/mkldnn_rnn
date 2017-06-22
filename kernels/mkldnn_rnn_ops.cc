/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifdef INTEL_MKL

#define EIGEN_USE_THREADS

#include <stddef.h>
#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <unordered_set>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"

/*
 * This module implements ops that fuse a multi-layer multi-step RNN/LSTM model
 * using the underlying Cudnn library.
 *
 * Cudnn RNN library exposes an opaque parameter buffer with unknown layout and
 * format. And it is very likely that if saved, they cannot be used across
 * different GPUs. So users need to first query the size of the opaque
 * parameter buffer, and convert it to and from its canonical forms. But each
 * actual training step is carried out with the parameter buffer.
 *
 * Similar to many other ops, the forward op has two flavors: training and
 * inference. When training is specified, additional data in reserve_space will
 * be produced for the backward pass. So there is a performance penalty.
 *
 * In addition to the actual data and reserve_space, Cudnn also needs more
 * memory as temporary workspace. The memory management to and from
 * stream-executor is done through ScratchAllocator. In general,
 * stream-executor is responsible for creating the memory of proper size. And
 * TensorFlow is responsible for making sure the memory is alive long enough
 * and recycles afterwards.
 *
 */
 

 
//#include "tensorflow/core/util/mkl_util.h" ? based on mkl_dnn.h
//#include "third_party/mkl/include/mkldnn.h"
#include "third_party/mkl/include/mkldnn.hpp"
#include "third_party/mkl/include/mkldnn_types.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

template <typename Device, typename T>
class MkldnnRNNForwardOp;

template <typename Device, typename T>
class MkldnnRNNBackwardOp;

using mkldnn::memory;
using mkldnn::algorithm;
using mkldnn::direction;
using mkldnn::input_mode;
using mkldnn::engine;
using mkldnn::prop_kind;
using mkldnn::stream;

struct MkldnnModelTypes {
  algorithm rnn_mode;
  input_mode rnn_input_mode;
  direction rnn_direction_mode;
  bool HasInputC() const {
    // only LSTM has input-c. All other models use only input-h.
    return rnn_mode == algorithm::rnn_lstm;
  }
};

// A helper class that collects the shapes to describe a RNN model.
struct MkldnnModelShapes {
  int num_layers;
  int input_size;
  int num_units;
  int seq_length;
  int batch_size;
  int dir_count;
  TensorShape input_shape;
  TensorShape output_shape;
  TensorShape hidden_state_shape;
  // At present only fields related to cached RnnDescriptor are concerned.
  bool IsCompatibleWith(const MkldnnModelShapes& rhs) const {
    return num_layers == rhs.num_layers && input_size == rhs.input_size &&
           num_units == rhs.num_units && dir_count == rhs.dir_count;
  }
  string RnnDescDebugString() {
    return strings::Printf(
        "[num_layers, input_size, num_units, dir_count]: [%d, %d, %d, %d]",
        num_layers, input_size, num_units, dir_count);
  }
};

// Extract and checks the forward input tensors, parameters, and shapes from the
// OpKernelContext.
Status ExtractForwardInput(OpKernelContext* context,
                           const MkldnnModelTypes& model_types,
                           const Tensor** input, const Tensor** input_h,
                           const Tensor** input_c, const Tensor** params,
                           MkldnnModelShapes* model_shapes) {
  TF_RETURN_IF_ERROR(context->input("input", input));
  TF_RETURN_IF_ERROR(context->input("input_h", input_h));
  if (model_types.HasInputC()) {
    TF_RETURN_IF_ERROR(context->input("input_c", input_c));
  }
  TF_RETURN_IF_ERROR(context->input("params", params));

  if ((*input)->dims() != 3) {
    return errors::InvalidArgument("RNN input must be a 3-D vector.");
  }
  model_shapes->seq_length = (*input)->dim_size(0);
  model_shapes->batch_size = (*input)->dim_size(1);
  model_shapes->input_size = (*input)->dim_size(2);
  model_shapes->input_shape = (*input)->shape();
  model_shapes->dir_count = (model_types.rnn_direction_mode == direction::rnn_bidirectional) ? 2 : 1;

  if ((*input_h)->dims() != 3) {
    return errors::InvalidArgument("RNN input must be a 3-D vector.");
  }
  model_shapes->num_layers = (*input_h)->dim_size(0) / model_shapes->dir_count;
  model_shapes->num_units = (*input_h)->dim_size(2);

  model_shapes->hidden_state_shape =
      TensorShape({model_shapes->dir_count * model_shapes->num_layers,
                   model_shapes->batch_size, model_shapes->num_units});
  if ((*input_h)->shape() != model_shapes->hidden_state_shape) {
    return errors::InvalidArgument(
        "Invalid input_h shape: ", (*input_h)->shape().DebugString(), " ",
        model_shapes->hidden_state_shape.DebugString());
  }
  if (model_types.HasInputC()) {
    if ((*input_h)->shape() != (*input_c)->shape()) {
      return errors::InvalidArgument(
          "input_h and input_c must have the same shape: ",
          (*input_h)->shape().DebugString(), " ",
          (*input_c)->shape().DebugString());
    }
  }
  model_shapes->output_shape =
      TensorShape({model_shapes->seq_length, model_shapes->batch_size,
                   model_shapes->dir_count * model_shapes->num_units});
  return Status::OK();
}


// A common base class for RNN kernels. It extracts common attributes and
// shape validations.
class MkldnnRNNKernelCommon : public OpKernel {
 protected:
  explicit MkldnnRNNKernelCommon(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("rnn_mode", &model_types_.rnn_mode));
    OP_REQUIRES_OK(context, context->GetAttr("input_mode", &model_types_.rnn_input_mode));
    OP_REQUIRES_OK(context, context->GetAttr("direction", &model_types_.rnn_direction_mode));
  }

  bool HasInputC() const { return model_types_.HasInputC(); }
  algorithm rnn_mode() const { return model_types_.rnn_mode; }
  input_mode rnn_input_mode() const { return model_types_.rnn_input_mode; }
  direction rnn_direction_mode() const {
    return model_types_.rnn_direction_mode;
  }
  MkldnnModelTypes model_types() const { return model_types_; }

 private:
  //  bool reset_rnd_gen_state_;  TBD
  MkldnnModelTypes model_types_;
};


// Run the forward operation of the RNN model.
template <typename T>
class MkldnnRNNForwardOp<CPUDevice, T> : public MkldnnRNNKernelCommon {
 public:
  typedef CPUDevice Device;
  explicit MkldnnRNNForwardOp(OpKernelConstruction* context)
      : MkldnnRNNKernelCommon(context) {
    OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));
  }

  void Compute(OpKernelContext* context) override {
    MkldnnRNNForwardOpContext mkl_context;
    const Tensor* input = nullptr;
    const Tensor* input_h = nullptr;
    const Tensor* input_c = nullptr;
    const Tensor* params = nullptr;
    MkldnnModelShapes model_shapes;

    OP_REQUIRES_OK(context,
                   ExtractForwardInput(context, &model_types_, &input, &input_h,
                                       &input_c, &params, &model_shapes));
    const auto& input_shape = model_shapes.input_shape;
    const auto& hidden_state_shape = model_shapes.hidden_state_shape;
    const auto& output_shape = model_shapes.output_shape;

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    Tensor* output_h = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, hidden_state_shape, &output_h));
    Tensor* output_c = nullptr;
    if (HasInputC()) {
      // Only LSTM uses input_c and output_c. So for all other models, we only
      // need to create dummy outputs.
      OP_REQUIRES_OK(
          context, context->allocate_output(2, hidden_state_shape, &output_c));
    } else {
      OP_REQUIRES_OK(context, context->allocate_output(2, {}, &output_c));
    }

    if (!is_training_) {
      Tensor* dummy_reserve_space = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(3, {}, &dummy_reserve_space));
    }
    // TBD, if there is workspace, need to allocate based on rnn ws desc
    std::shared_ptr<engine> eng;
    std::shared_ptr<memory> x;
    std::shared_ptr<memory> hx;
    std::shared_ptr<memory> cx;
    std::shared_ptr<memory> y;
    std::shared_ptr<memory> hy;
    std::shared_ptr<memory> cy;
    std::shared_ptr<memory> weights;
    std::shared_ptr<memory> workspace;
    std::shared_ptr<memory::desc> x_desc;
    std::shared_ptr<memory::desc> hx_desc;
    std::shared_ptr<memory::desc> y_desc;
    std::shared_ptr<memory::desc> weights_desc;
    std::shared_ptr<rnn_forward::primitive_desc> rnn_fwd_prim_desc;
    int state_outputs = 1; // for rnn_forward::desc creation

    data_type = memory::data_type::f32;
    eng.reset(new engine(engine::kind::cpu, 0));

    int w1_size = model_shapes->num_units * (model_shapes->num_units + model_shapes->input_size + 2);
    int wx_size = model_shapes->num_units * (model_shapes->num_units + model_shapes->num_units + 2);
    if (HasInputC()) {
      wl_size = wl_size * 4;
      wx_size = wx_size * 4;
    }
    const int total_w = model_shapes->num_layers == 1 ? model_shapes->dir_count * w1_size : model_shapes->dir_count
                        * (w1_size + (model_shapes->num_layers - 1) * wx_size);
 
    x_desc.reset(new memory::desc({model_shapes->seq_length,
                                   model_shapes->batch_size,
                                   model_shapes->input_size},
                                   data_type, memory::format::rnx));
    hx_desc.reset(new memory::desc({model_shapes->num_layers,
                                    model_shapes->batch_size,
                                    model_shapes->num_units},
                                    data_type, memory::format::rnx));
    y_desc.reset(new memory::desc({model_shapes->seq_length,
                                   model_shapes->batch_size,
                                   model_shapes->num_units * model_shapes->dir_count},
                                   data_type, memory::format::rnx));
    weights_desc.reset(new memory::desc({total_w}, data_type, memory::format::x));

    x.reset(new memory({ *x_desc, *eng }));
    hx.reset(new memory({ *hx_desc, *eng }));
    cx.reset(new memory({ *hx_desc, *eng }));
    y.reset(new memory({ *y_desc, *eng }));
    hy.reset(new memory({ *hx_desc, *eng }));
    cy.reset(new memory({ *hx_desc, *eng }));
    weights.reset(new memory({ *weights_desc, *eng }));
    
    prop_kind aprop_kind = is_training_ ? prop_kind::forward_training : prop_kind::forward_inference;
    auto rnn_fwd_desc = rnn_forward::desc(aprop_kind, model_types_.rnn_mode,
                model_types_.rnn_direction_mode, model_types_.rnn_input_mode, model_shapes->num_units,
                model_shapes->num_layers, model_shapes->seq_length,
                state_outputs, *x_desc, *hx_desc, *y_desc, *weights_desc);
    rnn_fwd_prim_desc.reset(new rnn_forward::primitive_desc(rnn_fwd_desc, *eng));

    std::vector<primitive> pipeline;
    auto s = stream(stream::kind::lazy);
    
    //TBD  x/hx/weights->get_primitive_desc().get_size() should be equal to that from tensor

    memcpy(x->get_data_handle(), input->template flat<T>().data(), input->template flat<T>().size() * sizeof(T));
    memcpy(hx->get_data_handle(), input_h->template flat<T>().data(), input_h->template flat<T>().size() * sizeof(T)); 
    memcpy(weights->get_data_handle(), params->template flat<T>().data(), params->template flat<T>().size() * sizeof(T)); 
    
    if (is_training_) {
      auto workspace_primitive_desc
                    = rnn_fwd_prim_desc->workspace_primitive_desc();
      workspace.reset(new memory(workspace_primitive_desc));
      // TBD: get workspace shape and creat output reserve space
      if (HasInputC()) {
        memcpy(cx->get_data_handle(), input_c->template flat<T>().data(), input_c->template flat<T>().size() * sizeof(T)); 
        auto l = rnn_forward(*rnn_fwd_prim_desc, x.get(), hx.get(), cx.get(),
                 weights.get(), y.get(), hy.get(), cy.get(), workspace.get());
        pipeline.push_back(l);
        s.submit(pipeline).wait();
        memcpy(output_c->template flat<T>().data(), cy->get_data_handle(), output_h->template flat<T>().size() * sizeof(T)); 
      } else {
        auto l = rnn_forward(*rnn_fwd_prim_desc, x.get(), hx.get(), nullptr, 
                 weights.get(), y.get(), hy.get(), nullptr, workspace.get());
        pipeline.push_back(l);
        s.submit(pipeline).wait();
      }
      // TBD need to copy workspace to output reserve_space
    } else {
      if (HasInputC()) {
        memcpy(cx->get_data_handle(), input_c->template flat<T>().data(), input_c->template flat<T>().size() * sizeof(T)); 
        auto l = rnn_forward(*rnn_fwd_prim_desc, x.get(), hx.get(), cx.get(),
                 weights.get(), y.get(), hy.get(), cy.get(), nullptr);
        pipeline.push_back(l);
        s.submit(pipeline).wait();
        memcpy(output_c->template flat<T>().data(), cy->get_data_handle(), output_h->template flat<T>().size() * sizeof(T)); 
      } else {
        auto l = rnn_forward(*rnn_fwd_prim_desc, x.get(), hx.get(), nullptr, 
                 weights.get(), y.get(), hy.get(), nullptr, nullptr);
        pipeline.push_back(l);
        s.submit(pipeline).wait();
      }
    }
    memcpy(output->template flat<T>().data(), y->get_data_handle(), output->template flat<T>().size() * sizeof(T));
    memcpy(output_h->template flat<T>().data(), hy->get_data_handle(), output_h->template flat<T>().size() * sizeof(T)); 
  }

 private:
  bool is_training_;
};

REGISTER_KERNEL_BUILDER(
    Name("MkldnnRNN").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    MkldnnRNNForwardOp<CPUDevice, float>);



// Run the backward operation of the RNN model.
template <typename T>
class MkldnnRNNBackwardOp<GPUDevice, T> : public MkldnnRNNKernelCommon {
 public:
  typedef CPUDevice Device;

  explicit MkldnnRNNBackwardOp(OpKernelConstruction* context)
      : MkldnnRNNKernelCommon(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* input = nullptr;
    const Tensor* input_h = nullptr;

    const Tensor* input_c = nullptr;
    const Tensor* params = nullptr;
    MkldnnModelShapes model_shapes;
    OP_REQUIRES_OK(context,
                   ExtractForwardInput(context, &model_types_, &input, &input_h,
                                       &input_c, &params, &model_shapes));

    const auto& input_shape = model_shapes.input_shape;
    const auto& hidden_state_shape = model_shapes.hidden_state_shape;
    const auto& output_shape = model_shapes.output_shape;

    const Tensor* reserve_space = nullptr;
    OP_REQUIRES_OK(context, context->input("reserve_space", &reserve_space));
    const Tensor* input_dy = nullptr;
    OP_REQUIRES_OK(context, context->input("input_dy", &input_dy));
    OP_REQUIRES(context, output_shape == input_dy->shape(),
                errors::InvalidArgument(
                    "input_h and input_c must have the same shape: ",
                    input_h->shape().DebugString(), " ",
                    input_c->shape().DebugString()));
    const Tensor* input_dhy = nullptr;
    OP_REQUIRES_OK(context, context->input("input_dhy", &input_dhy));
    OP_REQUIRES(context, input_dhy->shape() == hidden_state_shape,
                errors::InvalidArgument(
                    "Invalid input_dhy shape: ", input_dhy->shape().DebugString(),
                    " ", hidden_state_shape.DebugString()));
    const Tensor* input_dcy = nullptr;
    if (HasInputC()) {
      // Only LSTM uses input_c and output_c. So for all other models, we only
      // need to create dummy outputs.
      OP_REQUIRES_OK(context, context->input("input_dcy", &input_dcy));
      OP_REQUIRES(context, input_dcy->shape() == hidden_state_shape,
                  errors::InvalidArgument("Invalid input_dcy shape: ",
                                          input_dcy->shape().DebugString(), " ",
                                          hidden_state_shape.DebugString()));
    }
    Tensor* output_dx = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, input->shape(), &output_dx));
    Tensor* output_dhx = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, input_h->shape(),
                                                     &output_dhx));
    Tensor* output_dcx = nullptr;
    if (HasInputC()) {
      OP_REQUIRES_OK(context, context->allocate_output(2, input_c->shape(),
                                                       &output_dcx));
    } else {
      OP_REQUIRES_OK(context,
                     context->allocate_output(2, {}, &output_dcx));
    }
    Tensor* output_dweights = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(3, params->shape(),
                                                     &output_dweights));

    std::shared_ptr<engine> eng;
    std::shared_ptr<memory> x;
    std::shared_ptr<memory> hx;
    std::shared_ptr<memory> cx;
    std::shared_ptr<memory> y;
    std::shared_ptr<memory> hy;
    std::shared_ptr<memory> cy;
    std::shared_ptr<memory> dy;
    std::shared_ptr<memory> dhy;
    std::shared_ptr<memory> dcy;
    std::shared_ptr<memory> dx;
    std::shared_ptr<memory> dhx;
    std::shared_ptr<memory> dcx;
    std::shared_ptr<memory> weights;
    std::shared_ptr<memory> workspace;
    std::shared_ptr<memory> dweights;
    std::shared_ptr<memory::desc> x_desc;
    std::shared_ptr<memory::desc> hx_desc;
    std::shared_ptr<memory::desc> y_desc;
    std::shared_ptr<memory::desc> weights_desc;
    std::shared_ptr<rnn_forward::primitive_desc> rnn_fwd_prim_desc;
    std::shared_ptr<rnn_backward::primitive_desc> rnn_bwd_prim_desc;
    int state_outputs = 1; // for rnn_forward::desc creation

    data_type = memory::data_type::f32;
    eng.reset(new engine(engine::kind::cpu, 0));

    int w1_size = model_shapes->num_units * (model_shapes->num_units + model_shapes->input_size + 2);
    int wx_size = model_shapes->num_units * (model_shapes->num_units + model_shapes->num_units + 2);
    if (HasInputC()) {
      wl_size = wl_size * 4;
      wx_size = wx_size * 4;
    }
    const int total_w = model_shapes->num_layers == 1 ? model_shapes->dir_count * w1_size : model_shapes->dir_count
                        * (w1_size + (model_shapes->num_layers - 1) * wx_size);
 
    x_desc.reset(new memory::desc({model_shapes->seq_length,
                                   model_shapes->batch_size,
                                   model_shapes->input_size},
                                   data_type, memory::format::rnx));
    hx_desc.reset(new memory::desc({model_shapes->num_layers,
                                    model_shapes->batch_size,
                                    model_shapes->num_units},
                                    data_type, memory::format::rnx));
    y_desc.reset(new memory::desc({model_shapes->seq_length,
                                   model_shapes->batch_size,
                                   model_shapes->num_units * model_shapes->dir_count},
                                   data_type, memory::format::rnx));
    weights_desc.reset(new memory::desc({total_w}, data_type, memory::format::x));

    x.reset(new memory({ *x_desc, *eng }));
    hx.reset(new memory({ *hx_desc, *eng }));
    cx.reset(new memory({ *hx_desc, *eng }));
    y.reset(new memory({ *y_desc, *eng }));
    hy.reset(new memory({ *hx_desc, *eng }));
    cy.reset(new memory({ *hx_desc, *eng }));
    weights.reset(new memory({ *weights_desc, *eng }));
    dx.reset(new memory({ *x_desc, *eng }));
    dhx.reset(new memory({ *hx_desc, *eng }));
    dcx.reset(new memory({ *hx_desc, *eng }));
    dy.reset(new memory({ *y_desc, *eng }));
    dhy.reset(new memory({ *hx_desc, *eng }));
    dcy.reset(new memory({ *hx_desc, *eng }));
    dweights.reset(new memory({ *weights_desc, *eng }));
    
    prop_kind fwd_prop_kind = prop_kind::forward_training;
    auto rnn_fwd_desc = rnn_forward::desc(fwd_prop_kind, model_types_.rnn_mode,
                model_types_.rnn_direction_mode, model_types_.rnn_input_mode, model_shapes->num_units,
                model_shapes->num_layers, model_shapes->seq_length,
                state_outputs, *x_desc, *hx_desc, *y_desc, *weights_desc);
    rnn_fwd_prim_desc.reset(new rnn_forward::primitive_desc(rnn_fwd_desc, *eng));

    prop_kind bwd_prop_kind = prop_kind::backward;
    auto rnn_bwd_desc = rnn_backward::desc(bwd_prop_kind, model_types_.rnn_mode, 
                model_types_.rnn_direction_mode, model_types_.rnn_input_mode, model_shapes->num_units, 
                model_shapes->num_layers, model_shapes->seq_length, 
                state_outputs, *x_desc, *hx_desc, *y_desc, *weights_desc);
    rnn_bwd_prim_desc.reset(new rnn_backward::primitive_desc(
                rnn_bwd_desc, *eng, *rnn_fwd_prim_desc));

    auto workspace_primitive_desc  = rnn_fwd_prim_desc_->workspace_primitive_desc();
    workspace.reset(new memory(workspace_primitive_desc));

    std::vector<primitive> pipeline;
    auto s = stream(stream::kind::lazy);
    
    //TBD  x/hx/weights->get_primitive_desc().get_size() should be equal to that from tensor

    memcpy(x->get_data_handle(), input->template flat<T>().data(), input->template flat<T>().size() * sizeof(T));
    memcpy(hx->get_data_handle(), input_h->template flat<T>().data(), input_h->template flat<T>().size() * sizeof(T)); 
    memcpy(weights->get_data_handle(), params->template flat<T>().data(), params->template flat<T>().size() * sizeof(T)); 
    memcpy(dy->get_data_handle(), input_dy->template flat<T>().data(), input_dy->template flat<T>().size() * sizeof(T));
    memcpy(dhy->get_data_handle(), input_dhy->template flat<T>().data(), input_dhy->template flat<T>().size() * sizeof(T)); 
    memcpy(workspace->get_data_handle(), reserve_space->template flat<T>().data(), reserve_space->template flat<T>().size() * sizeof(T)); 

    // TBD: get workspace shape and creat output reserve space
    if (HasInputC()) {
      memcpy(cx->get_data_handle(), input_c->template flat<T>().data(), input_c->template flat<T>().size() * sizeof(T)); 
      memcpy(dcy->get_data_handle(), input_dcy->template flat<T>().data(), input_dcy->template flat<T>().size() * sizeof(T));

      auto l = rnn_backward(*rnn_bwd_prim_desc, x.get(), hx.get(), cx.get(),
                dy.get(), dhy.get(), dcy.get(), weights.get(), workspace.get(),
                dx.get(), dhx.get(), dcx.get(), dweights.get());
      pipeline.push_back(l);
      s.submit(pipeline).wait();
        
      memcpy(output_dcx->template flat<T>().data(), dcx->get_data_handle(), output_dcx->template flat<T>().size() * sizeof(T)); 

    } else {
      auto l = rnn_backward(*rnn_bwd_prim_desc, x.get(), hx.get(), nullptr,
                dy.get(), dhy.get(), nullptr, weights.get(), workspace.get(),
                dx.get(), dhx.get(), nullptr, dweights.get());
      pipeline.push_back(l);
      s.submit(pipeline).wait();
    }
          
    memcpy(output_dx->template flat<T>().data(), dx->get_data_handle(), output_dx->template flat<T>().size() * sizeof(T));
    memcpy(output_dhx->template flat<T>().data(), dhx->get_data_handle(), output_dhx->template flat<T>().size() * sizeof(T)); 
};

REGISTER_KERNEL_BUILDER(
    Name("MkldnnRNNBackprop").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    MkldnnRNNBackwardOp<CPUDevice, float>);


}  // namespace tensorflow

#endif  // INTEL_MKL