#include "kimera-vio/loopclosure/netvlad.h"

#include <torch/script.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <algorithm>
#include <utility>

namespace cpp_netvlad {

constexpr int FULL_VECTOR_SIZE = 32768;
constexpr auto DTYPE = at::kFloat;

NetVLAD::NetVLAD(std::string checkpoint_path): grad_guard_() {
  try {
    // Load TorchScript trace of PyTorch implementation of NetVLAD based on https://github.com/Nanne/pytorch-NetVlad
    script_net_ = torch::jit::load(checkpoint_path);
    script_net_.to(DTYPE); // conversion probably not necessary, but will avoid errors
  } catch (const c10::Error& e) {
    LOG(FATAL) << "Failed to load NetVLAD model.";
  }
  LOG(INFO) << "NetVLAD loaded successfully.";
}

void NetVLAD::transform(const cv::Mat& img, at::Tensor& rep) {
  // Assert that the returned tensor is normalized and detached
  std::vector<torch::jit::IValue> inputs;
  // Interprets the raw mono8 image data as a 1xRxC matrix of bytes (uint8)
  // Source: https://github.com/pytorch/pytorch/issues/12506#issuecomment-429573396
  at::Tensor tensor_img = torch::from_blob(img.data, {img.channels(), img.rows, img.cols}, at::TensorOptions(at::kByte));
  tensor_img = tensor_img.to(DTYPE); // Matching model
  // Expand to (1, 3, R, C) to pretend we have a color image with batch size 1.
  // This was included in the training/testing code, but I haven't seen examples of its use, so performance is unclear
  tensor_img = tensor_img.expand({1, 3, -1, -1});
  // Normalize the image based on constants from https://github.com/Nanne/pytorch-NetVlad/blob/master/pittsburgh.py input_transform()
  tensor_img = (tensor_img/255.f - at::tensor({0.485, 0.456, 0.406}).to(DTYPE).expand({1, 1, 1, 3}).permute({0, 3, 1, 2}))
                 / at::tensor({0.229, 0.224, 0.225}).to(DTYPE).expand({1, 1, 1, 3}).permute({0, 3, 1, 2});
  inputs.push_back(tensor_img);

  // Run NetVLAD
  at::Tensor output = script_net_.forward(inputs).toTensor();
  if(output.dim() != 2 || output.size(0) != 1 || output.size(1) != FULL_VECTOR_SIZE) {
    LOG(FATAL) << "NetVLAD output is not expected size: dim: " << output.dim() << " [" << output.size(0) << ", " << output.size(-1) << "]";
  }

  // TODO: determine an appropriate PCA reduction for output tensor to avoid storing 16MB per image

  rep = output.squeeze();
}
void NetVLAD::query(const at::Tensor& query, DBoW2::QueryResults& query_results,
                    const int max_results, const int max_id) const {
  // TODO: check if this is slow and maybe use a heap, since we have to significantly reduce the number of results while checking *all* images
  query_results.clear();
  query_results.reserve(database_.size());
  int i = 0;
  for(auto it = database_.begin(); it != database_.end(); it++, i++) {
    if(i >= max_id) {
      // Following example of DBoW2, we throw away all ids unless < max_id
      break;
    }
    query_results.emplace_back(i, score(query, *it));
  }
  std::sort(query_results.rbegin(), query_results.rend()); // Sort query_results in reverse so it ends up descending
  
  if(query_results.size() > max_results) {
    query_results.resize(max_results);
  }
}
void NetVLAD::add(const at::Tensor& rep) {
  database_.push_back(rep);
}
double NetVLAD::score(const at::Tensor& rep1, const at::Tensor& rep2) const {
  // Can assume that tensors are normalized, so Euclidean norm^2 is in [0, 2], with 0 best.
  // This score was designed to be interpreted via norm, so we return 1 - norm^2 / 2 to
  // follow convention of score in [0, 1] with 1 best
  if(rep1.size(0) == 0 || rep2.size(0) == 0) {
    // one rep is uninitialized, so the score should be bad
    return 0;
  } else if(rep1.size(0) != FULL_VECTOR_SIZE || rep2.size(0) != FULL_VECTOR_SIZE) {
    LOG(FATAL) << "NetVLAD::score had vector inputs of different sizes: " << rep1.size(0) << " " << rep2.size(0);
  }
  const double norm = (rep1 - rep2).norm().item<double>();
  if(norm > 2) {
    LOG(WARNING) << "NetVLAD::score had large norm: " << norm << " from inputs with norms: "
                 << rep1.norm().item<double>() << " " << rep2.norm().item<double>();
  }
  return std::max(1 - norm/2., 0.);
}

} // namespace cpp_netvlad