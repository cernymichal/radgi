#pragma once

#include <memory>
#include <utility>

template <typename T>
inline std::shared_ptr<T> copyShared(const std::shared_ptr<T>& ptr) {
    return std::make_shared<T>(*ptr);
}

/*
 * @brief A shared object.
 */
template <typename T>
using Ref = std::shared_ptr<T>;

template <typename T>
using WeakRef = std::weak_ptr<T>;

template <typename T, typename... Args>
inline Ref<T> makeRef(Args&&... args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
}

template <typename T, typename S>
inline Ref<T> castRef(const Ref<S>& ptr) {
    return std::dynamic_pointer_cast<T>(ptr);
}

template <typename T>
inline Ref<T> clone(const Ref<T>& ref) {
    return copyShared<T>(ref);
}

template <typename T, typename S>
inline Ref<T> cloneAs(const Ref<S>& ref) {
    return clone(castRef<T>(ref));
}

/*
 * @brief A single owner object.
 */
template <typename T>
using Scoped = std::unique_ptr<T>;

template <typename T, typename... Args>
inline Scoped<T> makeScoped(Args&&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}
