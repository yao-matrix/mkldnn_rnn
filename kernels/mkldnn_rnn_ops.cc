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
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/env_var.h"

#include "tensorflow/contrib/mkldnn_rnn/mkl-dnn/include/mkldnn.hpp"

/*
 * This module implements ops that fuse a multi-layer multi-step RNN/LSTM model
 * using the underlying Cudnn library.
 *
 * Similar to many other ops, the forward op has two flavors: training and
 * inference. When training is specified, additional data in reserve_space will
 * be produced for the backward pass. So there is a performance penalty.
 *
 */
namespace tensorflow {
using CPUDevice = Eigen::ThreadPoolDevice;

template <typename Device, typename T, typename Index>
class MkldnnRNNParamsSizeOp;

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
using mkldnn::error;
using mkldnn::rnn_forward;
using mkldnn::rnn_backward;
using mkldnn::primitive;

Status ParseRNNMode(const string& str, algorithm* rnn_mode) {
  if (str == "rnn_relu") {
    *rnn_mode = algorithm::rnn_relu;
    return Status::OK();
  } else if (str == "rnn_tanh") {
    *rnn_mode = algorithm::rnn_tanh;
    return Status::OK();
  } else if (str == "lstm") {
    *rnn_mode = algorithm::rnn_lstm;
    return Status::OK();
  } else if (str == "gru") {
    *rnn_mode = algorithm::rnn_gru;
    return Status::OK();
  }
  return errors::InvalidArgument("Invalid RNN mode: ", str);
}

Status ParseRNNInputMode(const string& str, input_mode* rnn_input_mode) {
  if (str == "linear_input") {
    *rnn_input_mode = input_mode::rnn_linear_input;
    return Status::OK();
  } else if (str == "skip_input") {
    *rnn_input_mode = input_mode::rnn_skip_input;
    return Status::OK();
  } else if (str == "auto_select") {
    *rnn_input_mode = input_mode::rnn_linear_input;
    return Status::OK();
  }
 
  return errors::InvalidArgument("Invalid RNN input mode: ", str);
}

Status ParseRNNDirectionMode(const string& str,
                             direction* rnn_dir_mode) {
  if (str == "unidirectional") {
    *rnn_dir_mode = direction::rnn_unidirectional;
    return Status::OK();
  } else if (str == "bidirectional") {
    *rnn_dir_mode = direction::rnn_bidirectional;
    return Status::OK();
  }
  return errors::InvalidArgument("Invalid RNN direction mode: ", str);
}

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

  // input layout: T x N x F
  model_shapes->seq_length = (*input)->dim_size(0);
  model_shapes->batch_size = (*input)->dim_size(1);
  model_shapes->input_size = (*input)->dim_size(2);
  model_shapes->input_shape = (*input)->shape();
  model_shapes->dir_count = (model_types.rnn_direction_mode == direction::rnn_bidirectional) ? 2 : 1;
  // LOG(ERROR) << "input size: " << (*input)->dim_size(0) << ", " << (*input)->dim_size(1) << ", " << (*input)->dim_size(2);

  if ((*input_h)->dims() != 3) {
    return errors::InvalidArgument("RNN input must be a 3-D vector.");
  }

  // h layout: (L * dir_count) x N x num_units
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

  // c layout: (L * dir_count) x N x num_units
  if (model_types.HasInputC()) {
    if ((*input_h)->shape() != (*input_c)->shape()) {
      return errors::InvalidArgument(
          "input_h and input_c must have the same shape: ",
          (*input_h)->shape().DebugString(), " ",
          (*input_c)->shape().DebugString());
    }
  }

  // output layout: T x N x (dir_count * num_units)
  model_shapes->output_shape =
      TensorShape({model_shapes->seq_length, model_shapes->batch_size,
                   model_shapes->dir_count * model_shapes->num_units});
  return Status::OK();
}


// A common base class for RNN kernels. It extracts common attributes and
// shape validations.
class MkldnnRNNKernelCommon : public OpKernel {
 protected:
  explicit MkldnnRNNKernelCommon(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dropout", &dropout_));
    OP_REQUIRES_OK(context, context->GetAttr("seed", &seed_));
    OP_REQUIRES_OK(context, context->GetAttr("seed2", &seed2_));
    string str;
    OP_REQUIRES_OK(context, context->GetAttr("rnn_mode", &str));
    OP_REQUIRES_OK(context, ParseRNNMode(str, &model_types_.rnn_mode));
    OP_REQUIRES_OK(context, context->GetAttr("input_mode", &str));
    OP_REQUIRES_OK(context, ParseRNNInputMode(str, &model_types_.rnn_input_mode));
    OP_REQUIRES_OK(context, context->GetAttr("direction", &str));
    OP_REQUIRES_OK(context, ParseRNNDirectionMode(str, &model_types_.rnn_direction_mode));
  }

  bool HasInputC() const { return model_types_.HasInputC(); }
  algorithm rnn_mode() const { return model_types_.rnn_mode; }
  input_mode rnn_input_mode() const { return model_types_.rnn_input_mode; }
  direction rnn_direction_mode() const {
    return model_types_.rnn_direction_mode;
  }
  MkldnnModelTypes model_types() const { return model_types_; }
  float dropout() const { return dropout_; }
  uint64 seed() { return (static_cast<uint64>(seed_) << 32) | seed2_; }
 private:
  int seed_;
  int seed2_;
  float dropout_;
  // bool reset_rnd_gen_state_;

  MkldnnModelTypes model_types_;
};

// A class that returns the size of the parameter buffer. The user should
// use that to create the actual parameter buffer for training. However, it
// should not be used for saving and restoring.
template <typename T, typename Index>
class MkldnnRNNParamsSizeOp<CPUDevice, T, Index> : public MkldnnRNNKernelCommon {
 public:
  typedef CPUDevice Device;
  explicit MkldnnRNNParamsSizeOp(OpKernelConstruction* context)
      : MkldnnRNNKernelCommon(context) {}

  void Compute(OpKernelContext* context) override {
    int64 params_size = -1;
    int dir_count = rnn_direction_mode() == direction::rnn_unidirectional ? 1 : 2;

    const Tensor* num_layers_t = nullptr;
    context->input("num_layers", &num_layers_t);
    if (!TensorShapeUtils::IsScalar(num_layers_t->shape())) {
      LOG(ERROR) << "num_layers is not a scalar";
    }
    int num_layers = num_layers_t->scalar<int>()();

    const Tensor* num_units_t = nullptr;
    context->input("num_units", &num_units_t);
    if (!TensorShapeUtils::IsScalar(num_units_t->shape())) {
      LOG(ERROR) << "num_units is not a scalar";
    }
    int num_units = num_units_t->scalar<int>()();

    const Tensor* input_size_t = nullptr;
    context->input("input_size", &input_size_t);
    if (!TensorShapeUtils::IsScalar(input_size_t->shape())) {
      LOG(ERROR) << "input_size is not a scalar";
    }
    int input_size = input_size_t->scalar<int>()();

    int first_layer_weights = 0;
    int higher_layer_weights = 0;
    int all_biases = 0;

    // TODO need complete logics here
    switch (rnn_mode()) {
      case algorithm::rnn_relu:
      case algorithm::rnn_tanh:
        first_layer_weights = num_units * (input_size + num_units + 2);
        higher_layer_weights = (num_layers - 1) * num_units * (2 * num_units + 2);
        params_size = (first_layer_weights + higher_layer_weights) * dir_count;
        break;
      case algorithm::rnn_lstm:
        first_layer_weights = 4 * num_units * (input_size + num_units + 2);
        higher_layer_weights = 4 * (num_layers - 1) * num_units * (2 * num_units + 2);
        params_size = (first_layer_weights + higher_layer_weights) * dir_count;
        break;
      case algorithm::rnn_gru:
        // TODO
        break;
      default:
        LOG(WARNING) << "Invalid RNN mode: " << rnn_mode();
        break;
    }

    Tensor* output_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {1}, &output_t));
    *output_t->template flat<Index>().data() = params_size;
  }
};

REGISTER_KERNEL_BUILDER(Name("MkldnnRNNParamsSize")
                            .Device(DEVICE_CPU).TypeConstraint<float>("T").TypeConstraint<int32>("S"),
                        MkldnnRNNParamsSizeOp<CPUDevice, float, int32>);

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
    const Tensor* Tx = nullptr;
    const Tensor* Thx = nullptr;
    const Tensor* Tcx = nullptr;
    const Tensor* Tweights = nullptr;
    MkldnnModelShapes model_shapes;

    OP_REQUIRES_OK(context,
                   ExtractForwardInput(context, model_types(), &Tx, &Thx,
                                       &Tcx, &Tweights, &model_shapes));
    // const auto& input_shape = model_shapes.input_shape;
    const auto& hidden_state_shape = model_shapes.hidden_state_shape;
    const auto& output_shape = model_shapes.output_shape;

    Tensor* Ty = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &Ty));
    Tensor* Thy = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, hidden_state_shape, &Thy));
    Tensor* Tcy = nullptr;
    if (HasInputC()) {
      // Only LSTM uses input_c and output_c. So for all other models, we only
      // need to create dummy outputs.
      OP_REQUIRES_OK(
          context, context->allocate_output(2, hidden_state_shape, &Tcy));
    } else {
      OP_REQUIRES_OK(context, context->allocate_output(2, {}, &Tcy));
    }

    Tensor* dummy_reserve_space = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(3, {}, &dummy_reserve_space));

    // FIXME if there is workspace, need to allocate based on rnn ws desc
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
    int wl_size;
    int wx_size;

    memory::data_type a_data_type = memory::data_type::f32;
    eng.reset(new engine(engine::kind::cpu, 0));

    wl_size = model_shapes.num_units * (model_shapes.num_units + model_shapes.input_size + 2);
    wx_size = model_shapes.num_units * (model_shapes.num_units + model_shapes.num_units + 2);
    if (HasInputC()) {
      wl_size = wl_size * 4;
      wx_size = wx_size * 4;
    }
    const int total_w = model_shapes.num_layers == 1 ? model_shapes.dir_count * wl_size : model_shapes.dir_count
                        * (wl_size + (model_shapes.num_layers - 1) * wx_size);

    x_desc.reset(new memory::desc({model_shapes.seq_length,
                                   model_shapes.batch_size,
                                   model_shapes.input_size},
                                   a_data_type, memory::format::rnx));
    hx_desc.reset(new memory::desc({model_shapes.num_layers,
                                    model_shapes.batch_size,
                                    model_shapes.num_units},
                                   a_data_type, memory::format::rnx));
    y_desc.reset(new memory::desc({model_shapes.seq_length,
                                   model_shapes.batch_size,
                                   model_shapes.num_units * model_shapes.dir_count},
                                   a_data_type, memory::format::rnx));
    weights_desc.reset(new memory::desc({total_w}, a_data_type, memory::format::x));

    x.reset(new memory({ *x_desc, *eng }));
    hx.reset(new memory({ *hx_desc, *eng }));
    cx.reset(new memory({ *hx_desc, *eng }));
    y.reset(new memory({ *y_desc, *eng }));
    hy.reset(new memory({ *hx_desc, *eng }));
    cy.reset(new memory({ *hx_desc, *eng }));
    weights.reset(new memory({ *weights_desc, *eng }));

    prop_kind a_prop_kind = is_training_ ? prop_kind::forward_training : prop_kind::forward_inference;
    auto rnn_fwd_desc = mkldnn::rnn_forward::desc(a_prop_kind, rnn_mode(),
                                                  rnn_direction_mode(), rnn_input_mode(), model_shapes.num_units,
                                                  model_shapes.num_layers, model_shapes.seq_length,
                                                  state_outputs, *x_desc, *hx_desc, *y_desc, *weights_desc);
    rnn_fwd_prim_desc.reset(new rnn_forward::primitive_desc(rnn_fwd_desc, *eng));

    std::vector<primitive> pipeline;
    auto s = stream(stream::kind::lazy);

    // FIXME  x/hx/weights->get_primitive_desc().get_size() should be equal to that from tensor

    memcpy(x->get_data_handle(), Tx->template flat<T>().data(), Tx->template flat<T>().size() * sizeof(T));
    memcpy(hx->get_data_handle(), Thx->template flat<T>().data(), Thx->template flat<T>().size() * sizeof(T));
    memcpy(weights->get_data_handle(), Tweights->template flat<T>().data(), Tweights->template flat<T>().size() * sizeof(T));

    if (is_training_) {
      auto workspace_primitive_desc = rnn_fwd_prim_desc->workspace_primitive_desc();
      workspace.reset(new memory(workspace_primitive_desc));
      // TODO get workspace shape and creat output reserve space
      if (HasInputC()) {
        memcpy(cx->get_data_handle(), Tcx->template flat<T>().data(), Tcx->template flat<T>().size() * sizeof(T));
        auto l = rnn_forward(*rnn_fwd_prim_desc, x.get(), hx.get(), cx.get(),
                 weights.get(), y.get(), hy.get(), cy.get(), workspace.get());
        pipeline.push_back(l);
        s.submit(pipeline).wait();
        memcpy(Tcy->template flat<T>().data(), cy->get_data_handle(), Thy->template flat<T>().size() * sizeof(T));
      } else {
        auto l = rnn_forward(*rnn_fwd_prim_desc, x.get(), hx.get(), nullptr,
                 weights.get(), y.get(), hy.get(), nullptr, workspace.get());
        pipeline.push_back(l);
        s.submit(pipeline).wait();
      }
      // TODO need to copy workspace to output reserve_space
    } else {
      if (HasInputC()) {
        memcpy(cx->get_data_handle(), Tcx->template flat<T>().data(), Tcx->template flat<T>().size() * sizeof(T));
        auto l = rnn_forward(*rnn_fwd_prim_desc, x.get(), hx.get(), cx.get(),
                 weights.get(), y.get(), hy.get(), cy.get(), nullptr);
        pipeline.push_back(l);
        s.submit(pipeline).wait();
        memcpy(Tcy->template flat<T>().data(), cy->get_data_handle(), Thy->template flat<T>().size() * sizeof(T));
      } else {
        auto l = rnn_forward(*rnn_fwd_prim_desc, x.get(), hx.get(), nullptr,
                 weights.get(), y.get(), hy.get(), nullptr, nullptr);
        pipeline.push_back(l);
        s.submit(pipeline).wait();
      }
    }
    memcpy(Ty->template flat<T>().data(), y->get_data_handle(), Ty->template flat<T>().size() * sizeof(T));
    memcpy(Thy->template flat<T>().data(), hy->get_data_handle(), Thy->template flat<T>().size() * sizeof(T));
  }

 private:
  bool is_training_;
};

REGISTER_KERNEL_BUILDER(
    Name("MkldnnRNN").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    MkldnnRNNForwardOp<CPUDevice, float>);


// Run the backward operation of the RNN model.
template <typename T>
class MkldnnRNNBackwardOp<CPUDevice, T> : public MkldnnRNNKernelCommon {
 public:
  typedef CPUDevice Device;

  explicit MkldnnRNNBackwardOp(OpKernelConstruction* context)
      : MkldnnRNNKernelCommon(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* Tx = nullptr;
    const Tensor* Th = nullptr;

    const Tensor* Tc = nullptr;
    const Tensor* Tweights = nullptr;
    MkldnnModelShapes model_shapes;
    OP_REQUIRES_OK(context,
                   ExtractForwardInput(context, model_types(), &Tx, &Th,
                                       &Tc, &Tweights, &model_shapes));

    // const auto& input_shape = model_shapes.input_shape;
    const auto& hidden_state_shape = model_shapes.hidden_state_shape;
    const auto& output_shape = model_shapes.output_shape;

    const Tensor* Tworkspace = nullptr;
    OP_REQUIRES_OK(context, context->input("reserve_space", &Tworkspace));
    const Tensor* Tdy = nullptr;
    OP_REQUIRES_OK(context, context->input("output_backprop", &Tdy));
    OP_REQUIRES(context, output_shape == Tdy->shape(),
                errors::InvalidArgument(
                    "h and c must have the same shape: ",
                    Th->shape().DebugString(), " ",
                    Tc->shape().DebugString()));
    const Tensor* Tdhy = nullptr;
    OP_REQUIRES_OK(context, context->input("output_h_backprop", &Tdhy));
    OP_REQUIRES(context, Tdhy->shape() == hidden_state_shape,
                errors::InvalidArgument(
                    "Invalid dhy shape: ", Tdhy->shape().DebugString(),
                    " ", hidden_state_shape.DebugString()));
    const Tensor* Tdcy = nullptr;
    if (HasInputC()) {
      // Only LSTM uses input_c and output_c. So for all other models, we only
      // need to create dummy outputs.
      OP_REQUIRES_OK(context, context->input("output_c_backprop", &Tdcy));
      OP_REQUIRES(context, Tdcy->shape() == hidden_state_shape,
                  errors::InvalidArgument("Invalid dcy shape: ",
                                          Tdcy->shape().DebugString(), " ",
                                          hidden_state_shape.DebugString()));
    }
    Tensor* Tdx = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, Tx->shape(), &Tdx));
    Tensor* Tdhx = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, Th->shape(),
                                                     &Tdhx));
    Tensor* Tdcx = nullptr;
    if (HasInputC()) {
      OP_REQUIRES_OK(context, context->allocate_output(2, Tc->shape(),
                                                       &Tdcx));
    } else {
      OP_REQUIRES_OK(context,
                     context->allocate_output(2, {}, &Tdcx));
    }
    Tensor* Tdweights = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(3, Tweights->shape(),
                                                     &Tdweights));

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
    int wl_size;
    int wx_size;

    memory::data_type a_data_type = memory::data_type::f32;
    eng.reset(new engine(engine::kind::cpu, 0));

    wl_size = model_shapes.num_units * (model_shapes.num_units + model_shapes.input_size + 2);
    wx_size = model_shapes.num_units * (model_shapes.num_units + model_shapes.num_units + 2);
    if (HasInputC()) {
      wl_size = wl_size * 4;
      wx_size = wx_size * 4;
    }
    const int total_w = model_shapes.num_layers == 1 ? model_shapes.dir_count * wl_size : model_shapes.dir_count
                        * (wl_size + (model_shapes.num_layers - 1) * wx_size);
 
    x_desc.reset(new memory::desc({model_shapes.seq_length,
                                   model_shapes.batch_size,
                                   model_shapes.input_size},
                                   a_data_type, memory::format::rnx));
    hx_desc.reset(new memory::desc({model_shapes.num_layers,
                                    model_shapes.batch_size,
                                    model_shapes.num_units},
                                    a_data_type, memory::format::rnx));
    y_desc.reset(new memory::desc({model_shapes.seq_length,
                                   model_shapes.batch_size,
                                   model_shapes.num_units * model_shapes.dir_count},
                                   a_data_type, memory::format::rnx));
    weights_desc.reset(new memory::desc({total_w}, a_data_type, memory::format::x));

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

    auto rnn_fwd_desc = rnn_forward::desc(prop_kind::forward_training, 
                                          static_cast<mkldnn::algorithm>(rnn_mode()),
                                          static_cast<mkldnn::direction>(rnn_direction_mode()),
                                          static_cast<mkldnn::input_mode>(rnn_input_mode()),
                                          model_shapes.num_units, model_shapes.num_layers, model_shapes.seq_length,
                                          state_outputs, *x_desc, *hx_desc, *y_desc, *weights_desc);
    rnn_fwd_prim_desc.reset(new rnn_forward::primitive_desc(rnn_fwd_desc, *eng));

    auto rnn_bwd_desc = rnn_backward::desc(prop_kind::backward, 
                                           static_cast<mkldnn::algorithm>(rnn_mode()),
                                           static_cast<mkldnn::direction>(rnn_direction_mode()),
                                           static_cast<mkldnn::input_mode>(rnn_input_mode()),
                                           model_shapes.num_units, model_shapes.num_layers, model_shapes.seq_length,
                                           state_outputs, *x_desc, *hx_desc, *y_desc, *weights_desc);
    rnn_bwd_prim_desc.reset(new rnn_backward::primitive_desc(rnn_bwd_desc, *eng, *rnn_fwd_prim_desc));

    auto workspace_primitive_desc  = rnn_fwd_prim_desc->workspace_primitive_desc();
    workspace.reset(new memory(workspace_primitive_desc));

    std::vector<primitive> pipeline;
    auto s = stream(stream::kind::lazy);
    
    // TODO x/hx/weights->get_primitive_desc().get_size() should be equal to that from tensor

    memcpy(x->get_data_handle(), Tx->template flat<T>().data(), Tx->template flat<T>().size() * sizeof(T));
    memcpy(hx->get_data_handle(), Th->template flat<T>().data(), Th->template flat<T>().size() * sizeof(T));
    memcpy(weights->get_data_handle(), Tweights->template flat<T>().data(), Tweights->template flat<T>().size() * sizeof(T));
    memcpy(dy->get_data_handle(), Tdy->template flat<T>().data(), Tdy->template flat<T>().size() * sizeof(T));
    memcpy(dhy->get_data_handle(), Tdhy->template flat<T>().data(), Tdhy->template flat<T>().size() * sizeof(T));
    memcpy(workspace->get_data_handle(), Tworkspace->template flat<T>().data(), Tworkspace->template flat<T>().size() * sizeof(T));

    // TODO get workspace shape and creat output reserve space
    if (HasInputC()) {
      memcpy(cx->get_data_handle(), Tc->template flat<T>().data(), Tc->template flat<T>().size() * sizeof(T));
      memcpy(dcy->get_data_handle(), Tdcy->template flat<T>().data(), Tdcy->template flat<T>().size() * sizeof(T));

      auto l = rnn_backward(*rnn_bwd_prim_desc, x.get(), hx.get(), cx.get(),
                dy.get(), dhy.get(), dcy.get(), weights.get(), workspace.get(),
                dx.get(), dhx.get(), dcx.get(), dweights.get());
      pipeline.push_back(l);
      s.submit(pipeline).wait();

      memcpy(Tdcx->template flat<T>().data(), dcx->get_data_handle(), Tdcx->template flat<T>().size() * sizeof(T));
    } else {
      auto l = rnn_backward(*rnn_bwd_prim_desc, x.get(), hx.get(), nullptr,
                            dy.get(), dhy.get(), nullptr, weights.get(), workspace.get(),
                            dx.get(), dhx.get(), nullptr, dweights.get());
      pipeline.push_back(l);
      s.submit(pipeline).wait();
    }

    memcpy(Tdx->template flat<T>().data(), dx->get_data_handle(), Tdx->template flat<T>().size() * sizeof(T));
    memcpy(Tdhx->template flat<T>().data(), dhx->get_data_handle(), Tdhx->template flat<T>().size() * sizeof(T));
    memcpy(Tdweights->template flat<T>().data(), dweights->get_data_handle(), Tweights->template flat<T>().size() * sizeof(T));
  }
};

REGISTER_KERNEL_BUILDER(
    Name("MkldnnRNNBackprop").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    MkldnnRNNBackwardOp<CPUDevice, float>);
}  // namespace tensorflow

#endif  // INTEL_MKL
