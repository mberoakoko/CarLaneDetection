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
}

#endif //UTILS_HPP
