//
// Created by mbero on 29/01/2025.
//

#ifndef CARLANEDETECTION_HISTOGRAM_HPP
#define CARLANEDETECTION_HISTOGRAM_HPP
#include <bits/stdc++.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
namespace Utils{
    constexpr int HIST_BINS = 256;
    constexpr int HIST_WIDTH = 512;
    constexpr int HIST_HEIGHT = 400;

    struct RGB_Tag{};
    struct HSV_Tag{};
    struct HSL_Tag{};

    enum class ImageType: std::uint8_t {
        RGB = 0, HSV = 1, HSL = 2, YUV = 3
    };

    struct Channel_Color {
        cv::Scalar channel_1_color;
        cv::Scalar channel_2_color;
        cv::Scalar channel_3_color;
    };

   inline auto image_type_to_color_codes(const ImageType& type) -> Channel_Color {
       switch (type) {
           case ImageType::RGB: {
               return {
                   .channel_1_color = cv::Scalar(255, 0, 255),  // Red
                   .channel_2_color = cv::Scalar(0, 255, 255),  // Green
                   .channel_3_color = cv::Scalar(0, 0, 255)     // Blue
               };
           }
           case ImageType::HSV: {
               return {
                   .channel_1_color = cv::Scalar(255, 255, 255),  // White
                   .channel_2_color = cv::Scalar(241, 245, 196),  // Yellow
                   .channel_3_color = cv::Scalar(124, 169, 247)     // Light Blue
               };
           }
           case ImageType::HSL: {
               return {
                   .channel_1_color = cv::Scalar(255, 255, 255), // White
                   .channel_2_color = cv::Scalar(241, 245, 196), // Yellow
                   .channel_3_color = cv::Scalar(216, 142, 250)    // Light Pink
               };
           }
           case ImageType::YUV: {
               return {
                   .channel_1_color = cv::Scalar(255, 255, 255), // White
                   .channel_2_color = cv::Scalar(241, 245, 196), // Yellow
                   .channel_3_color = cv::Scalar(216, 142, 250)    // Light Pink
               };
           }
           default: {
               return {
                   .channel_1_color = cv::Scalar(255, 0, 255),
                   .channel_2_color = cv::Scalar(0, 255, 255),
                   .channel_3_color = cv::Scalar(0, 0, 255)
               };
           }
       }
   };

    struct hist_config {
        int hist_bins = HIST_BINS;
        int hist_w = HIST_WIDTH;
        int hist_h = HIST_HEIGHT;
        ImageType image_hist_type = ImageType::RGB;
    };




    class Histogram{
    public:
        explicit Histogram(hist_config&& config)
        :   bins_(config.hist_bins),
            hist_w_(config.hist_w),
            hist_h_(config.hist_h),
            image_type_(config.image_hist_type)
        {

        }

        template<typename T>
        auto generate(cv::Mat& input_image) const -> cv::Mat {
            return input_image.clone();
        }

        auto generate(cv::Mat& input_image) const -> cv::Mat{
            std::vector<cv::Mat> brg_channels;
            cv::split(input_image, brg_channels);
            float  range [] = {0, 256};
            const float* histRange[] = { range };
            bool uniform = true, accumulate = false;

            cv::Mat b_hist, g_hist, r_hist;
            cv::calcHist(&brg_channels.at(0), 1, 0, cv::Mat(), b_hist, 1, &bins_, histRange, uniform, accumulate);
            cv::calcHist(&brg_channels.at(1), 1, 0, cv::Mat(), g_hist, 1, &bins_, histRange, uniform, accumulate);
            cv::calcHist(&brg_channels.at(2), 1, 0, cv::Mat(), r_hist, 1, &bins_, histRange, uniform, accumulate);

            int bin_w = cvRound((double ) hist_w_/bins_);
            cv::Mat hist_image{hist_w_, hist_h_, CV_8UC3, cv::Scalar (0, 0, 0)};
            cv::normalize(b_hist, b_hist, 0, hist_image.rows,
                          cv::NORM_MINMAX, -1, cv::Mat() );
            cv::normalize(g_hist, g_hist, 0, hist_image.rows,
                          cv::NORM_MINMAX, -1, cv::Mat() );
            cv::normalize(r_hist, r_hist, 0, hist_image.rows,
                          cv::NORM_MINMAX, -1, cv::Mat() );

            for( int i = 1; i < bins_; i++ ) {
                // B hist line
                line(hist_image, cv::Point(bin_w * (i - 1), bins_ - cvRound(b_hist.at<float>(i - 1))),
                     cv::Point(bin_w * (i), hist_h_ - cvRound(b_hist.at<float>(i))),
                     cv::Scalar(255, 0, 0), 2, 8, 0);

                // R hist line
                line(hist_image, cv::Point(bin_w * (i - 1), bins_ - cvRound(g_hist.at<float>(i - 1))),
                     cv::Point(bin_w * (i), hist_h_ - cvRound(g_hist.at<float>(i))),
                     cv::Scalar(0, 255, 0), 2, 8, 0);

                // G hist line
                line(hist_image, cv::Point(bin_w * (i - 1), bins_ - cvRound(r_hist.at<float>(i - 1))),
                     cv::Point(bin_w * (i), hist_h_ - cvRound(r_hist.at<float>(i))),
                     cv::Scalar(0, 0, 255), 2, 8, 0);
            }
            return hist_image;
        }

        auto generate_histogram(const cv::Mat& input_image, std::optional<Utils::ImageType> image_type = std::nullopt)  -> cv::Mat {
            image_type_ = image_type == std::nullopt ? ImageType::RGB : image_type.value();
            cv::Mat hist_image{hist_w_, hist_h_, CV_8UC3, cv::Scalar (0, 0, 0)};
            cv::Mat converted_image = Histogram::convert_image_to(input_image, image_type_);
            std::vector<cv::Mat> channels { image_to_channel_split(converted_image) };
            std::vector<cv::Mat> histograms { channel_wise_histogram(channels) };
            hist_image = draw_histogram(hist_image, histograms);
            return hist_image;
        }

    private:
        static auto convert_image_to(const cv::Mat& input_image, Utils::ImageType type = Utils::ImageType::RGB) -> cv::Mat {
            switch (type) {
                case Utils::ImageType::RGB: {
                    cv::Mat output_image;
                    cv::cvtColor(input_image, output_image, cv::COLOR_BGR2RGB);
                    return output_image;
                }break;
                case Utils::ImageType::HSV: {
                        cv::Mat output_image;
                        cv::cvtColor(input_image, output_image, cv::COLOR_BGR2HSV);
                        return output_image;
                }break;
                case Utils::ImageType::HSL: {
                    cv::Mat output_image;
                    cv::cvtColor(input_image, output_image, cv::COLOR_BGR2HSV);
                    return output_image;
                }break;
                case ImageType::YUV: {
                        cv::Mat output_image;
                        cv::cvtColor(input_image, output_image, cv::COLOR_BGR2YUV);
                        return output_image;
                }
            }
            return input_image;
        }

        static auto image_to_channel_split(const cv::Mat& input_image) -> std::vector<cv::Mat> {
            std::vector<cv::Mat> channels;
            cv::split(input_image, channels);
            return  channels;
        }
         auto channel_wise_histogram(const std::vector<cv::Mat>& channels) const -> std::vector<cv::Mat> {
             float  range [] = {0, 256};
             const float* histRange[] = { range };
             bool uniform = true, accumulate = false;
            cv::Mat channel_1_hist, channel_2_hist, channel_3_hist;
             int bin_w = cvRound((double ) hist_w_/bins_);
             const cv::Mat hist_image{hist_w_, hist_h_, CV_8UC3, cv::Scalar (0, 0, 0)};
            cv::calcHist(&channels.at(0), 1, 0, cv::Mat(), channel_1_hist, 1, &bins_, histRange, uniform, accumulate);
            cv::calcHist(&channels.at(1), 1, 0, cv::Mat(), channel_2_hist, 1, &bins_, histRange, uniform, accumulate);
            cv::calcHist(&channels.at(2), 1, 0, cv::Mat(), channel_3_hist, 1, &bins_, histRange, uniform, accumulate);

             cv::normalize(channel_1_hist, channel_1_hist, 0, hist_image.rows,
                          cv::NORM_MINMAX, -1, cv::Mat() );
             cv::normalize(channel_2_hist, channel_2_hist, 0, hist_image.rows,
                           cv::NORM_MINMAX, -1, cv::Mat() );
             cv::normalize(channel_3_hist, channel_3_hist, 0, hist_image.rows,
                           cv::NORM_MINMAX, -1, cv::Mat() );
             return {channel_1_hist, channel_2_hist, channel_3_hist};
        }

        auto draw_histogram(const cv::Mat& hist_image, const std::vector<cv::Mat>& channels) const -> cv::Mat {
            int bin_w = cvRound(static_cast<double>(hist_w_)/bins_);
            Channel_Color channel_color { image_type_to_color_codes(image_type_) };
            for( int i = 1; i < bins_; i++ ) {
                // B hist line
                line(hist_image, cv::Point(bin_w * (i - 1), bins_ - cvRound(channels.at(0).at<float>(i - 1))),
                     cv::Point(bin_w * (i), hist_h_ - cvRound(channels.at(0).at<float>(i))),
                     channel_color.channel_1_color, 2, 8, 0);
                //
                // // R hist line
                line(hist_image, cv::Point(bin_w * (i - 1), bins_ - cvRound(channels.at(1).at<float>(i - 1))),
                     cv::Point(bin_w * (i), hist_h_ - cvRound(channels.at(1).at<float>(i))),
                     channel_color.channel_2_color, 2, 8, 0);
                //
                // // G hist line
                line(hist_image, cv::Point(bin_w * (i - 1), bins_ - cvRound(channels.at(2).at<float>(i - 1))),
                     cv::Point(bin_w * (i), hist_h_ - cvRound(channels.at(2).at<float>(i))),
                     channel_color.channel_3_color, 2, 8, 0);
            }
            return hist_image;
        }

    private:
        int bins_;
        int hist_w_;
        int hist_h_;
        ImageType image_type_;
    };
}

#endif //CARLANEDETECTION_HISTOGRAM_HPP
