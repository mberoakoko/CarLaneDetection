//
// Created by mbero on 19/05/2025.
//
#include <bits/stdc++.h>
#ifndef PIPELINES_HPP
#define PIPELINES_HPP
#include <bits/stdc++.h>
#include <opencv2/imgproc.hpp>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

namespace PipeLine {
    class ILaneDetectionPipeline {
    public:
        virtual ~ILaneDetectionPipeline() = default;
        virtual auto execute(const cv::Mat& input_image) -> cv::Mat = 0;
    };

    class IEdgeDetection {
    public:
        virtual ~IEdgeDetection() = default;
        virtual auto detect_edges(const cv::Mat& input_image) -> cv::Mat = 0;
    };

    class GradientMagnitudeEdgeDetection : public IEdgeDetection {
    public:
        explicit GradientMagnitudeEdgeDetection(std::uint8_t kernel): IEdgeDetection(), kernel_(kernel) {
        }

        auto detect_edges(const cv::Mat &input_image) -> cv::Mat override {
            cv::Mat result;
            cv::Sobel(input_image, grad_x_, CV_32F, 1, 0, kernel_);
            cv::Sobel(input_image, grad_y_, CV_32F, 0, 1, kernel_);
            abs_grad_x_ = cv::abs(grad_x_);
            abs_grad_y_ = cv::abs(grad_y_);
            cv::normalize(abs_grad_x_, abs_grad_x_, 0, 255, cv::NORM_MINMAX);
            cv::normalize(abs_grad_y_, abs_grad_y_, 0, 255, cv::NORM_MINMAX);

            cv::addWeighted(abs_grad_x_, 0.5, abs_grad_y_, 0.5, 0, grad_);
            cv::normalize(grad_, grad_, 0, 255, cv::NORM_MINMAX);
            cv::threshold(grad_, result, 65, 255, cv::THRESH_BINARY);
            return result;
        };

    private:
        cv::Mat grad_x_, grad_y_;
        cv::Mat abs_grad_x_, abs_grad_y_;
        cv::Mat grad_;
        std::uint8_t kernel_;
    };

    class CannyEdgeDetection : public IEdgeDetection {
        public:
        explicit CannyEdgeDetection(std::uint8_t low_threshold, std::uint8_t ratio, std::uint8_t kernel_size)
        : IEdgeDetection(), low_threshold_(low_threshold), ratio_(ratio), kernel_size_(kernel_size) {}

        auto detect_edges(const cv::Mat &input_image) -> cv::Mat override {
            cv::Mat result;
            return result;
        };
    private:
        std::uint8_t low_threshold_;
        std::uint8_t ratio_;;
        std::uint8_t kernel_size_;
    };

    class GradientFilterPipeline : public ILaneDetectionPipeline {
    public:
        auto execute(const cv::Mat &input_image) -> cv::Mat override;
    };

    class CannyGaussianFilterPipeline: public ILaneDetectionPipeline {
    public:
        auto execute(const cv::Mat &input_image) -> cv::Mat override;
    };


    class SlidingWindowPipeline final: public ILaneDetectionPipeline {
    public:
        struct HistogramEdges {
            cv::Mat histogram_data;
            std::uint16_t left_edge;
            std::uint16_t right_edge;
        };

        static auto calculate_histogram_edges(const cv::Mat &input_image) -> HistogramEdges {
            std::uint16_t mid_point_0_axis = ::round(input_image.rows/2);
            std::uint16_t mid_point_1_axis = ::round(input_image.cols/2);
            cv::Mat lower_region = input_image(
                cv::Range(mid_point_0_axis, input_image.rows), cv::Range::all()
                );
            cv::Mat hist_data ;
            // std::reduce(lower_region, hist_data, 0, cv::REDUCE_SUM);
            auto min_along_axis = [](const cv::Mat& data_mat) -> std::uint16_t {
                cv::Point min_loc;
                cv::minMaxLoc(data_mat, nullptr, nullptr, &min_loc, nullptr);
                return static_cast<std::uint16_t>(min_loc.x);
            };
            return {
                .histogram_data = hist_data,
                .left_edge = min_along_axis(hist_data(cv::Range(0, mid_point_0_axis), cv::Range::all())),
                .right_edge = min_along_axis(hist_data(cv::Range(mid_point_0_axis, hist_data.rows), cv::Range::all()))
            };
        };

        auto execute(const cv::Mat &input_image) -> cv::Mat override {
            auto [histogram_data, left_edge, right_edge] =
                    SlidingWindowPipeline::calculate_histogram_edges(input_image);
            return input_image.clone();
        };
    };

    // TODO: use tag dispatching to enable the dispatching of various lane detection pipelines

    class PrototypePipeline: public ILaneDetectionPipeline {
    public:
        explicit PrototypePipeline(std::unique_ptr<IEdgeDetection>detector)
            :ILaneDetectionPipeline(),
            detector_(std::move(detector)) {

        }

    public:
        auto execute(const cv::Mat &input_image) -> cv::Mat override {
            cv::Mat result = detector_->detect_edges(input_image);
            return result;
        }

    private:
        std::unique_ptr<IEdgeDetection> detector_;
    };
}

#endif //PIPELINES_HPP
