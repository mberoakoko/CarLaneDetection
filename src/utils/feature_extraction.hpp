//
// Created by mbero on 30/01/2025.
//

#ifndef CARLANEDETECTION_FEATURE_EXTRACTION_HPP
#define CARLANEDETECTION_FEATURE_EXTRACTION_HPP
#include <bits/stdc++.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace FeatureExtraction{
    class SurfFeatureExtractor{

    public:
        explicit SurfFeatureExtractor() = default;
        ~SurfFeatureExtractor()= default;

        auto detect(cv::Mat& source){
            extractor_->detect(source, key_points_);
        }

        auto detect_and_draw(cv::Mat& source) -> cv::Mat{
            this->detect(source);
            cv::drawKeypoints(source, key_points_, extracted_image);
            return extracted_image;
        }
    private:
        cv::Ptr<cv::GFTTDetector> extractor_ = cv::GFTTDetector::create();
        std::vector<cv::KeyPoint> key_points_ {};
        cv::Mat extracted_image;
    };

    class ImagePreprocessor {
    public:
        enum class ExposureCorrectnessStrat: std::uint8_t {
            PIX_VAL_NORMAL, PIX_VAL_EXP_CORRECT
        };
        struct SharpenImageParams {
            double sigma;
            int threshold_value ;
            int amount ;
        };
        struct WhiteBalanceParams {
            double ratio_s;
            double ratio_l;
        };
        struct HighlightRemoverParams {
            std::uint8_t lightness_threshold { 255 };
            std::uint8_t saturation_threshold { 255 };
        };
        struct ShadowRemoverParams {
            std::uint8_t shadow_threshold { 100 };
        };
        struct ImageProcessorParams {
            double exposure_correctness_ratio = 1;
            ExposureCorrectnessStrat exposure_correctness_strat = ExposureCorrectnessStrat::PIX_VAL_NORMAL;
            SharpenImageParams sharpen_params {.sigma = 1, .threshold_value = 5, .amount = 1};
            WhiteBalanceParams white_balance_params = {.ratio_s = 1, .ratio_l = 1};
            HighlightRemoverParams highlight_remover_params = {.lightness_threshold = 1, .saturation_threshold = 100};
            ShadowRemoverParams shadow_remover_params = {.shadow_threshold = 255};

        };
        explicit ImagePreprocessor() = default;
        explicit ImagePreprocessor(const ImageProcessorParams &params) :
        exposure_correctness_ratio(params.exposure_correctness_ratio),
        ration_s(params.white_balance_params.ratio_s),
        ration_l(params.white_balance_params.ratio_l),
        sigma(params.sharpen_params.amount),
        threshold_value(params.sharpen_params.threshold_value),
        amount(params.sharpen_params.amount),
        lightness_threshold(params.highlight_remover_params.lightness_threshold),
        saturation_threshold(params.highlight_remover_params.saturation_threshold),
        shadow_threshold(params.shadow_remover_params.shadow_threshold){
        };
        ~ImagePreprocessor() = default;

        auto operator()(cv::Mat& source) -> cv::Mat {
            cv::Mat local_matrix ;;
            cv::cvtColor(source, local_matrix, cv::COLOR_RGB2HLS);
            local_matrix = shadow_remover(local_matrix);
            local_matrix = exposure_balance(local_matrix);
            local_matrix = white_balance(local_matrix);
            local_matrix = highlight_remover(local_matrix);
            local_matrix = shadow_remover(local_matrix);
            return local_matrix;
        };

    private:
        auto sharpen_image(cv::Mat& hls_frame) const -> cv::Mat {
            cv::Mat blurred;
            cv::GaussianBlur(hls_frame, blurred, cv::Size(blurred.cols, blurred.rows), sigma, amount);
            cv::Mat low_contrast_matrix = cv::abs(hls_frame -  blurred) < threshold_value;
            cv::Mat sharpened_image = hls_frame * (1 + amount) - blurred * (-amount);
            hls_frame.copyTo(sharpened_image, low_contrast_matrix);
            return sharpened_image;
        }
        auto exposure_balance(cv::Mat& hls_frame) -> cv::Mat {
            double minVal, maxVal;
            std::vector<cv::Mat> channels;
            cv::split(hls_frame, channels);
            auto l_channel = channels[1];
            cv::Scalar mean_pixel_value { cv::mean(l_channel) };;
            cv::minMaxLoc(l_channel, &minVal, &maxVal);

            // normalization strategies
            auto pix_val_exposure_correct =  [&](auto pix_value) {
                auto result = (pix_value - minVal) / ((maxVal - minVal) * (pix_value - mean_pixel_value.val[0]) * exposure_correctness_ratio);
                return result;
            };

            auto pix_val_normal_mean = [&](auto pix_value) {
                return (pix_value - minVal) / (maxVal - minVal) * 255.0;
            };


            switch (exposure_correctness_strat_) {
                case ExposureCorrectnessStrat::PIX_VAL_NORMAL:
                    l_channel = (l_channel - minVal) * (255 / (maxVal - minVal));
                    break;
                case ExposureCorrectnessStrat::PIX_VAL_EXP_CORRECT:
                    std::transform(l_channel.begin<double>(), l_channel.end<double>(),l_channel.begin<double>(), pix_val_exposure_correct);
                    break;
            }

            return hls_frame;

        };
        auto white_balance(cv::Mat& hls_frame) const -> cv::Mat {
            std::vector<cv::Mat> channels {};
            cv::split(hls_frame, channels);
            cv::Mat s_channel = channels[1];
            cv::Mat l_channel_negative = 255.0 - channels[2];
            cv::Mat frame_s_ln = (ration_s * s_channel) + (ration_l * l_channel_negative);
            double min_frame_s_ln, max_frame_s_ln;
            double temp_frame_min, temp_frame_max;
            cv::minMaxLoc(frame_s_ln, &min_frame_s_ln, &max_frame_s_ln);

            cv::Mat mask_min_s_ln = frame_s_ln == min_frame_s_ln;
            cv::Mat temp_frame_l_negative_buffer;
            l_channel_negative.copyTo(temp_frame_l_negative_buffer, mask_min_s_ln);
            cv::minMaxLoc(temp_frame_l_negative_buffer, &temp_frame_min, &temp_frame_max);
            cv::Mat mask_max_s_ln = l_channel_negative > temp_frame_min;
            cv::Mat min_s;
            s_channel.copyTo(min_s, mask_min_s_ln & mask_max_s_ln);
            s_channel = s_channel - min_s;
            cv::Mat less_than_zero_mask = s_channel < 0;
            s_channel.setTo(0, less_than_zero_mask);
            cv::Mat result ;
            cv::merge(channels, result);
            return result ;
        };
        auto highlight_remover(cv::Mat& hls_frame) -> cv::Mat {
            std::vector<cv::Mat> channels {};
            cv::split(hls_frame, channels);
            cv::Mat lightness_condition_mask = channels[1] > lightness_threshold;
            cv::Mat hue_condition_mask = channels[2] < saturation_threshold;
            cv::Mat composed_condition_mask = (lightness_condition_mask & hue_condition_mask);
            hls_frame.setTo(cv::Scalar(0, 0, 0), composed_condition_mask);
            return hls_frame;
        };
        auto shadow_remover(cv::Mat& hls_frame) -> cv::Mat {
            std::vector<cv::Mat> channels {};
            cv::split(hls_frame, channels);
            cv::Mat shadow_threshold_mask = channels[1] < shadow_threshold ;
            cv::Mat hue_condition_mask = channels[2] > saturation_threshold;
            cv::Mat composed_condition_mask = (hue_condition_mask & shadow_threshold);
            hls_frame.setTo(cv::Scalar(0, 0, 0), composed_condition_mask);
            return hls_frame;
        };
    private:
        double exposure_correctness_ratio = 1.0;
        ExposureCorrectnessStrat exposure_correctness_strat_ = ExposureCorrectnessStrat::PIX_VAL_NORMAL;
        double ration_s = 0.1;
        double ration_l = 0.9;
        double sigma = 1;
        int threshold_value = 5;
        int amount = 1;
        std::uint8_t lightness_threshold = 100;
        std::uint8_t saturation_threshold = 100;
        std::uint8_t shadow_threshold = 100;
    };

    class PerspectiveTransformer{ // Should Notify the two pipelines gradient filter and canny gaussian
    public:
        struct transformation_points{
            cv::Point2f top_left;
            cv::Point2f bottom_left;
            cv::Point2f top_right;
            cv::Point2f bottom_right;
        };
        explicit PerspectiveTransformer()= default;
        explicit PerspectiveTransformer(transformation_points& transformationPoints): points_(transformationPoints){
            points_s1_ = {points_.top_left, points_.bottom_left, points_.top_right, points_.bottom_right};
        }
//        explicit PerspectiveTransformer(std::initializer_list<transformation_points> tranformation_points):

        auto set_window_dimension(cv::Rect2i & rect) -> void {
            window_dimention = rect;
            points_2 = transformation_points{
                    .top_left = cv::Point2f (0, 0),
                    .bottom_left = cv::Point2f (0, 480),
                    .top_right = cv::Point2f (640, 0),
                    .bottom_right = cv::Point2f (640, 480)
            };
            points_s2_ = {points_2.top_left, points_2.bottom_left, points_2.top_right, points_2.bottom_right};
            perspective_transform_ = cv::getPerspectiveTransform(points_s1_, points_s2_,cv::DECOMP_QR);
        }

        auto get_perspective_markers(const cv::Mat& mat){
            mat.copyTo(perspective_markers_);
            cv::circle(perspective_markers_, points_.top_left, 5, cv::Scalar (255, 100, 0), 3);
            cv::circle(perspective_markers_, points_.bottom_left, 5, cv::Scalar (255, 255, 0), 4);
            cv::circle(perspective_markers_, points_.top_right, 5, cv::Scalar (255, 100, 255), 5);
            cv::circle(perspective_markers_, points_.bottom_right, 5, cv::Scalar (0, 255, 0), 6);
            return perspective_markers_;

        }

        /**
         * This Function creates a warped perspective of the dashboard camera.
         * we also store a copy of the warped perspective in this object,
         * @param mat Dashboard camera image
         * @return Warped Perspective of the Dashboard Camera
         */
        auto get_transformation_frame(const cv::Mat& mat) -> cv::Mat{
            cv::warpPerspective(mat, transformation_frame_, perspective_transform_, cv::Size(points_2.bottom_right) );
            return  transformation_frame_;
        }

    private:
        transformation_points points_, points_2;
        cv::Mat perspective_markers_, transformation_frame_, perspective_transform_;
        std::vector<cv::Point2f> points_s1_, points_s2_;
        cv::Rect2i window_dimention;
    };
}
#endif //CARLANEDETECTION_FEATURE_EXTRACTION_HPP
