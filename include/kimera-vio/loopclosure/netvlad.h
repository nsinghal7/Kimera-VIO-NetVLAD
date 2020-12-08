#pragma once

#include <utility>
#include <torch/torch.h>
#include <opencv2/core/core.hpp>
#include <DBoW2/QueryResults.h>
#include <map>


namespace cpp_netvlad {

class NetVLAD {
private:
 std::map<unsigned int, at::Tensor> database_;
 torch::jit::script::Module script_net_;
 torch::NoGradGuard grad_guard_;

public:
 NetVLAD(std::string checkpoint_path);
 void transform(const cv::Mat& img, at::Tensor& rep);
 void query(const at::Tensor& query, DBoW2::QueryResults& query_results,
            const int max_results, const int max_id) const;
 void add(const at::Tensor& rep, const unsigned int id);
 double score(const at::Tensor& rep1, const at::Tensor& rep2) const;
};

} // namespace cpp_netvlad
