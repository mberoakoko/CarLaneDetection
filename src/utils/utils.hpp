//
// Created by mbero on 19/05/2025.
//

#include <bits/stdc++.h>
#ifndef UTILS_HPP
#define UTILS_HPP

namespace utils {
    template<typename T>
    struct  Observer {
        virtual ~Observer() = default;
        virtual auto field_changed(T& source) -> void = 0;
    };

    template<typename T>
    class  Observable final {
    public:
        virtual ~Observable() = default;

        void notify(T& source) {
            for (const auto observer: observers) {
                observer->field_changed(source);
            }
        }

        void subscribe(std::unique_ptr<Observer<T>> observer) {
            observers.push_back(std::move(observer));
        }

        void unsubscribe(std::unique_ptr<Observer<T>> observer) {
            observers.erase(std::remove(observer.begin(), observer.end(), observer), observer.end());
        }
    private:
        std::vector<std::unique_ptr<Observer<T>>> observers;
    };


    namespace display_utils {
        struct pipeline_results {
            cv::Mat main;
            cv::Mat histogram;
            cv::Mat perspective;
            cv::Mat transformed;
            cv::Mat proto;
        };


        inline auto create_dashboard(const pipeline_results& pipe_results) -> cv::Mat {
            int tile_w = 400;
            int tile_h = 300;

            cv::Mat canvas = cv::Mat::zeros(tile_h * 2, tile_w * 3, CV_8UC3);

            auto draw_tile = [&](const cv::Mat& img, const int row, const int col, const std::string &label) -> void  {
                if (img.empty()) return;

                cv::Rect region_of_interest(col * tile_w, row * tile_h, tile_w, tile_h);

                cv::Mat resized;
                cv::resize(img, resized, cv::Size(tile_w, tile_h));

                resized.copyTo(canvas(region_of_interest));

                // 4. Add a label so you know what's what
                cv::putText(canvas, label, {region_of_interest.x + 10, region_of_interest.y + 25},
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2);
            };

            draw_tile(pipe_results.main,        0, 0, "Main");
            draw_tile(pipe_results.histogram,   0, 1, "Histogram");
            draw_tile(pipe_results.proto,       0, 2, "Proto Detection");
            draw_tile(pipe_results.perspective, 1, 0, "Perspective View");
            draw_tile(pipe_results.transformed, 1, 1, "Transformed");

            return canvas;
        }
    }
}

#endif //UTILS_HPP
