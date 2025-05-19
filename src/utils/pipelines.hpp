//
// Created by mbero on 19/05/2025.
//
#include <bits/stdc++.h>
#ifndef PIPELINES_HPP
#define PIPELINES_HPP
#include <bits/stdc++.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

namespace PipeLine {
    class lane_detection_pipeline {
        virtual ~lane_detection_pipeline() = default;
        virtual auto execute(const cv::Mat& input_image) -> cv::Mat = 0;
    };

    struct GradientFilterPipelineTag{};
    struct CannyGaussianFilterPipelineTag{};

    class GradientFilterPipeline : public lane_detection_pipeline {
        auto execute(const cv::Mat &input_image) -> cv::Mat override;
    };

    class CannyGaussianFilterPipeline: public lane_detection_pipeline {
        auto execute(const cv::Mat &input_image) -> cv::Mat override;
    };

    // TODO: use tag dispatching to enable the dispatching of various lane detection pipelines
}

#endif //PIPELINES_HPP
