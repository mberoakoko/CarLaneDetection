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

            if (source.empty()) { return source;}

            const cv::Mat hls = [&source] -> cv::Mat {
                cv::Mat local_matrix;
                cv::cvtColor(source, local_matrix, cv::COLOR_RGB2HLS);
                return local_matrix;
            }();


            const cv::Mat pipeline_result = [&hls, this] {
                cv::Mat temp = shadow_remover(hls);
                temp = exposure_balance(temp);
                temp = white_balance(temp);
                temp = highlight_remover(temp);
                temp = shadow_remover(temp);
                return temp;
            }();

            return pipeline_result;
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
            cv::Mat l_channel ;
            channels[1].convertTo(l_channel, CV_32F);

            cv::Scalar mean_pixel_value { cv::mean(l_channel) };
            cv::minMaxLoc(l_channel, &minVal, &maxVal);

            if (std::abs(maxVal - minVal) < 1e-5) return hls_frame;

            switch (exposure_correctness_strat_) {
                case ExposureCorrectnessStrat::PIX_VAL_NORMAL:
                    l_channel = (l_channel - minVal) * (255 / (maxVal - minVal));
                    break;
                case ExposureCorrectnessStrat::PIX_VAL_EXP_CORRECT:
                    cv::Mat denominator = (maxVal - minVal) * (l_channel - mean_pixel_value) * exposure_correctness_ratio;
                    cv::divide((l_channel - minVal), denominator, l_channel);
                    break;
            }

            l_channel.convertTo(channels[1], CV_8U);

            cv::Mat result;
            cv::merge(channels, result);
            return result;

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
        auto highlight_remover(const cv::Mat& hls_frame) const -> cv::Mat {
            std::vector<cv::Mat> channels {};
            cv::split(hls_frame, channels);
            cv::Mat lightness_condition_mask = channels[1] > lightness_threshold;
            cv::Mat hue_condition_mask = channels[2] < saturation_threshold;
            cv::Mat composed_condition_mask = (lightness_condition_mask & hue_condition_mask);

            cv::Mat result = hls_frame.clone();
            result.setTo(cv::Scalar(0, 0, 0), composed_condition_mask);
            return result;
        };
        auto shadow_remover(const cv::Mat& hls_frame) const -> cv::Mat {
            std::vector<cv::Mat> channels {};
            cv::split(hls_frame, channels);
            cv::Mat shadow_threshold_mask = channels[1] < shadow_threshold ;
            cv::Mat hue_condition_mask = channels[2] > saturation_threshold;
            cv::Mat composed_condition_mask = (hue_condition_mask & shadow_threshold_mask);
            cv::Mat result = hls_frame.clone();
            result.setTo(cv::Scalar(0, 0, 0), composed_condition_mask);
            return result;
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


    namespace perspective_transformation {
        class IPerspectiveTransformer {
        public:
            virtual ~IPerspectiveTransformer() = default;
            virtual auto get_perspective_markers(const cv::Mat& mat ) -> cv::Mat;
            virtual auto get_transformation_points(const cv::Mat& mat ) -> cv::Mat;
        };


        class HughTranformConsensus {
        public:

            struct CannyEdgeDetectionConfig {
                cv::Size blur_size ;
                double threshold;
                double ration;
                int arpeture_size;

                auto upper_val() const -> double { return  threshold * ration; }
            };

            struct ConfigHughlineP {
                double rho = 1.0;
                double theta = CV_PI/180;
                int threshold = 80;
                double mininum_line_length = 10.0;
            };

            using LinesArray = std::vector<cv::Vec4i>;
        private:
            /**
             * Simply Detect edges
             * @param image
             * @param config
             * @return
             */
            static auto canny_edge_detector(const cv::Mat& image ,
                                            const CannyEdgeDetectionConfig config = {
                                                .blur_size = cv::Size{3, 3},
                                                .threshold = 10,
                                                .ration = 10,
                                                .arpeture_size = 3
                                            }) {
                cv::Mat result;
                cv::blur(image, result, config.blur_size);
                cv::Canny(result, result, config.threshold, config.threshold * config.ration, config.arpeture_size);
                return result;
            }


            /**
             * Simpy perform hugh transform
             * @param image
             * @param config
             * @return
             */
            static auto hugh_transformation(
                const cv::Mat& image,
                const ConfigHughlineP config = {
                .rho = 1.0,
                .theta = CV_PI/180,
                .threshold = 80,
                .mininum_line_length = 10.0,
            }) -> LinesArray{
                LinesArray lines;
                cv::HoughLinesP(image, lines, config.rho, config.theta, config.mininum_line_length, config.threshold);
                return  lines;
            }

            auto line_detection(const cv::Mat& image) -> LinesArray {
                if (images_.size() < max_cache_size_) {
                    images_.push_front(image);
                } else {
                    images_.pop_back();
                    images_.push_front(image);
                };
                std::vector<LinesArray> lines_result;
                std::transform(images_.begin(), images_.end(), lines_result.begin(),
                    [this](const cv::Mat& image) {
                            cv::Mat result = canny_edge_detector(image, this->config_.canny_config);
                            return HughTranformConsensus::hugh_transformation(image, this->config_.hugh_line_config);
                    });

                const auto flattened_view = lines_result | std::ranges::views::join;
                return LinesArray{flattened_view.begin(), flattened_view.end()};

            }

            auto filter_horizantal_lines(const LinesArray& lines) const -> LinesArray {
                auto is_horizantal = [this](const cv::Vec4i& line) -> bool {
                    auto theta = line[1];
                    float targetTheta = CV_PI / 2.0; // 90 degrees
                    float deltaRad = this->config_.line_filter_config.delta_deg * (CV_PI / 180.0);
                    return std::abs(theta - targetTheta) < deltaRad;
                };
                auto filter_view = lines | std::views::filter(is_horizantal);
                return LinesArray{filter_view.begin(), filter_view.end()};
            }

            static auto get_intersection(const cv::Vec4i& line_1, const cv::Vec4i& line_2) -> IntersectionResultType {
                const float x1 = line_1[0], y1 = line_1[1], x2 = line_1[2], y2 = line_1[3];
                const float x3 = line_2[0], y3 = line_2[1], x4 = line_2[2], y4 = line_2[3];

                const float denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

                // The "Parallel" Failure Case
                if (std::abs(denom) < 1e-6f) {
                    return std::nullopt; // The "Bad" Track
                }

                cv::Point2f pt;
                pt.x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom;
                pt.y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom;

                return pt;
            }

            static auto find_vanishing_point(const std::vector<cv::Point2f>& intersections) -> cv::Point2f {
                if (intersections.empty()) return cv::Point2f(0, 0);

                cv::Point2f sum(0, 0);
                for (const auto& pt : intersections) {
                    sum += pt;
                }

                return cv::Point2f(sum.x / intersections.size(), sum.y / intersections.size());
            }

            auto get_vanishing_point(const cv::Mat& image) -> cv::Point2f {
                LinesArray lines = filter_horizantal_lines(line_detection(image));
                auto left_batch = lines | std::views::drop(1);
                auto right_batch = lines | std::views::take(lines.size() - 1);
                auto zipped = std::views::zip(left_batch, right_batch);
                auto intersections = zipped | std::views::transform(get_intersection);
                // filter then reduce with find vanishing point
            }

        public:
            struct LineTransfromConfig {
                CannyEdgeDetectionConfig canny_config;
                ConfigHughlineP hugh_line_config;
            };
            explicit HughTranformConsensus(std::deque<cv::Mat>& images, int max_cache_size, const LineTransfromConfig& config)
            :images_(images),
            max_cache_size_(max_cache_size),
            config_(config) {

            }



        private:
            std::deque<cv::Mat>& images_;
            const LineTransfromConfig config_;
            int max_cache_size_;

        };

        class VanishingPointCalibration final: public IPerspectiveTransformer {
        private:
            std::stack<cv::Mat> image_cache_;

        public:
            struct VanishingPoint {
                cv::Point2f px;
                cv::Point2f py;
            };

            explicit VanishingPointCalibration() = default;



        };
    }




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
