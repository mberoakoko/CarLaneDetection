#include <iostream>
#include <bits/stdc++.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"

#include "src/utils/histogram.hpp"
#include "src/utils/feature_extraction.hpp"
#include "src/utils/pipelines.hpp"
#include <ranges>
#include <iterator>

using rv = std::ranges::view_base;

namespace PipeLine{
    auto equalize_multi_channel(cv::Mat& input_image) -> cv::Mat{
        std::vector<cv::Mat> brg_channels;
        std::vector<cv::Mat> transformed_channels;
        cv::Mat result;
        cv::split(input_image, brg_channels);
        std::ranges::transform(brg_channels, std::back_inserter(transformed_channels), [](cv::Mat & mat){
            cv::equalizeHist(mat, mat);
            return mat;
        });
        cv::merge(transformed_channels, result);
        return result;
    }
}

class WebCamStream{
public:
    explicit WebCamStream(): prototype_(PipeLine::PrototypePipeline(
            std::make_unique<PipeLine::GradientMagnitudeEdgeDetection>(3)
        )){
        cap_.open("../data/Highway_5_low_sun.mp4");
        if (!cap_.isOpened()) {
            std::cout << "Something went terribly wrong" << std::endl;
        }
        cv::namedWindow("Main", cv::WINDOW_AUTOSIZE);
        // cv::namedWindow("Tertiary", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("PerspectiveView", cv::WINDOW_FULLSCREEN);
        cv::namedWindow("PerspectiveTransformation", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Histogram", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("ProtoDetection", cv::WINDOW_AUTOSIZE);

        tf = this->initialize_transform_points();
        perspectiveTransformer = FeatureExtraction::PerspectiveTransformer{tf};
        const auto rect = cv::Rect2i(0, 0, 0, 0);
        perspectiveTransformer.set_window_dimension(const_cast<cv::Rect2i &>(rect));
    }

    ~WebCamStream(){
        cap_.release();
        cv::destroyAllWindows();
    }
    auto exec() -> void{
        while (true){
            cap_.read(buffer);
            buffer.copyTo(processed);
            histogram_.generate_histogram(buffer, {Utils::ImageType::YUV}).copyTo(hist_img_);

            // buffer = PipeLine::equalize_multi_channel(buffer);
            buffer = perspectiveTransformer.get_perspective_markers(buffer);
            auto perspective_image = perspectiveTransformer.get_transformation_frame(buffer);
            auto transformed_image = image_preprocessor(perspective_image);
            auto prototype_image = prototype_.execute(
                [&] {
                    cv::Mat result;
                    cv::cvtColor(transformed_image, result, cv::COLOR_BGR2RGB);
                  return  result;
                }()
                );
            // cv::cvtColor(processed, gray_scale, cv::COLOR_BGR2GRAY);
//            cv::threshold(gray_scale, gray_scale, 200, 250, cv::THRESH_BINARY);
            // cv::adaptiveThreshold(gray_scale, gray_scale,255, cv::ADAPTIVE_THRESH_MEAN_C,cv::THRESH_BINARY,11, 10);
//            auto size = cv::Size (1, 2);
            // cv::medianBlur(gray_scale, gray_scale, 1);

            // auto gray_scale_feature_detection = extractor.detect_and_draw(gray_scale)

            cv::imshow("Main",buffer);
            // cv::imshow("Tertiary",gray_scale_feature_detection);
            cv::imshow("Histogram", hist_img_);
            cv::imshow("PerspectiveView", perspective_image);
            cv::imshow("PerspectiveTransformation", transformed_image);
            cv::imshow("ProtoDetection", prototype_image);
            if (cv::waitKey(10) == 27){
                break;
            }
            if (buffer.empty()){
                break;
            }
        }
    }

private:
    auto initialize_transform_points() -> FeatureExtraction::PerspectiveTransformer::transformation_points{
        auto window_rect = cv::getWindowImageRect("Main");
        using PerspectiveTranformer = FeatureExtraction::PerspectiveTransformer;
        PerspectiveTranformer ::transformation_points transform_points{
                .top_left=cv::Point2f ((window_rect.width/2) - 200, (window_rect.height/2) + 200),
                .bottom_left=cv::Point2f (window_rect.width/2 - 550, window_rect.height),
                .top_right=cv::Point2f ((window_rect.width/2) + 120, (window_rect.height/2) +200),
                .bottom_right=cv::Point2f (window_rect.width/2 + 400, window_rect.height)
        };
        return transform_points;
    }

    cv::VideoCapture cap_;
    cv::Mat buffer;
    cv::Mat processed;
    cv::Mat gray_scale;

    Utils::Histogram histogram_{Utils::hist_config{}};
    cv::Mat hist_img_;

    FeatureExtraction::SurfFeatureExtractor extractor{};
    FeatureExtraction::PerspectiveTransformer::transformation_points tf;
    FeatureExtraction::PerspectiveTransformer perspectiveTransformer {};
    FeatureExtraction::ImagePreprocessor image_preprocessor {};
    PipeLine::PrototypePipeline prototype_;
};

template<typename  T>
class fib_iter{
    T a = 0, b = 1;
public:
    using iterator_category = std::input_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = const T*;
    using reference = const T &;

    auto operator++() -> fib_iter<T>{
        T temp = a;
        a = b;
        b += temp;
        return *this;
    }
    auto operator*() -> T{
        return a;
    }
    auto operator != (const fib_iter<T> fib_iter) const -> bool {return true;}
};

struct  my_record{
    int a{0};
    int b{0};
    ~my_record(){
        std::cout<<"Destructor called"<<std::endl;
    }
};
auto create_record() -> std::unique_ptr<my_record>{
    auto record = std::make_unique<my_record>();
    return record;
}
void do_something_with_record(const std::unique_ptr<my_record> record){ // resource sink
    std::cout<<record->b<<std::endl;
}

template<typename Predicate, typename Func>
auto transform_if(Predicate pred, Func func) {
    return std::views::filter(pred) | std::views::transform(func);
}
int main() {
    WebCamStream webCamStream{};
    webCamStream.exec();

}
