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

    class PerspectiveTransformer{
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

        auto get_transformation_frame(cv::Mat& mat) -> cv::Mat{
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
