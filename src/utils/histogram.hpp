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

    class Histogram{
    public:
        struct hist_config {
            int hist_bins = HIST_BINS;
            int hist_w = HIST_WIDTH;
            int hist_h = HIST_HEIGHT;
        };

        explicit Histogram(hist_config&& config)
        :   bins_(config.hist_bins),
            hist_w_(config.hist_w),
            hist_h_(config.hist_h){

        }

        auto generate(cv::Mat& input_image) -> cv::Mat{
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


    private:
        int bins_;
        int hist_w_;
        int hist_h_;
    };
}

#endif //CARLANEDETECTION_HISTOGRAM_HPP
