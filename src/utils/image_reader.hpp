//
// Created by mbero on 06/06/2025.
//

#ifndef IMAGE_READER_HPP
#define IMAGE_READER_HPP
#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
namespace ImageReader {

    // ██╗███╗   ███╗ █████╗  ██████╗ ███████╗    ██████╗ ███████╗ █████╗ ██████╗ ███████╗██████╗
    // ██║████╗ ████║██╔══██╗██╔════╝ ██╔════╝    ██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔════╝██╔══██╗
    // ██║██╔████╔██║███████║██║  ███╗█████╗      ██████╔╝█████╗  ███████║██║  ██║█████╗  ██████╔╝
    // ██║██║╚██╔╝██║██╔══██║██║   ██║██╔══╝      ██╔══██╗██╔══╝  ██╔══██║██║  ██║██╔══╝  ██╔══██╗
    // ██║██║ ╚═╝ ██║██║  ██║╚██████╔╝███████╗    ██║  ██║███████╗██║  ██║██████╔╝███████╗██║  ██║
    // ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝
    //

    // The purpose of this namespace is to design an idiomatic method to read images from disk
    // all while handling erros elegantly
    enum class VisionError {
        CAP_NOT_AVAILABLE, IMAGE_NOT_AVAILABLE
    };

    template<typename T >
    class VisionResult {
    public:
        static auto success(T val) -> VisionResult {
            return VisionResult{ std::move(val) };
        }
        static auto failure(const VisionError& error) -> VisionResult {
            return  VisionResult{error};
        }

        using ValueType = T;

        template<typename Func>
        auto and_then(Func&& func) && -> decltype(auto) {
            if (value_.has_value()) {
                return VisionResult::success(std::forward<Func>(func)(std::move(value_).value()));
            }
            return VisionResult::failure(error_);
        }

        template<typename  Func, typename Arg>
        auto operator >>(Func && func) && -> VisionResult<Arg> {
            return std::move(this).and_then(std::forward<Func>(func));
        }

        [[nodiscard]]
        auto is_success() const -> bool { return value_.has_value(); }

        [[nodiscard]]
        auto is_failure() const -> bool { return error_.has_value(); }

        [[nodiscard]]
        auto get_error() const -> VisionError { return error_.value(); }


    private:
        std::optional<T> value_;
        std::optional<VisionError> error_;
        explicit VisionResult(T value) : value_(std::move(value)) {}
        explicit VisionResult(const VisionError& error) : error_(error) {}
    };


    inline auto get_capture(const std::string& path) -> VisionResult<cv::VideoCapture> {
        cv::VideoCapture cap;
        cap.open(path);
        if (!cap.isOpened()) {
            return VisionResult<cv::VideoCapture>::failure(VisionError::CAP_NOT_AVAILABLE);
        }
        return VisionResult<cv::VideoCapture>::success(cap);
    }

    inline auto grey_scale_convert(const cv::Mat& image) -> VisionResult<cv::Mat> {
        if (image.empty()) {
            std::cerr << "Image is empty." << std::endl;
            return VisionResult<cv::Mat>::failure(VisionError::IMAGE_NOT_AVAILABLE);
        }
        cv::Mat result;
        cv::cvtColor(image, result, cv::COLOR_BGR2RGB);
        return VisionResult<cv::Mat>::success(result);
    }


}
#endif //IMAGE_READER_HPP
