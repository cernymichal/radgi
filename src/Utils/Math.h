#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <ostream>

constexpr double PI = glm::pi<double>();
constexpr double TWO_PI = glm::two_pi<double>();
constexpr double HALF_PI = glm::half_pi<double>();

using glm::dvec2;
using glm::dvec3;
using glm::dvec4;
using glm::ivec2;
using glm::ivec3;
using glm::ivec4;
using glm::mat2;
using glm::mat3;
using glm::mat4;
using glm::quat;
using glm::uvec2;
using glm::uvec3;
using glm::uvec4;
using glm::vec2;
using glm::vec3;
using glm::vec4;

/*
 * @param v The incoming vector
 * @param normal The normal vector
 */
constexpr inline vec3 reflect(const vec3& v, const vec3& normal) {
    return v - 2.0f * glm::dot(v, normal) * normal;
}

/*
 * @param v The incoming vector, normalized
 * @param normal The normal vector, normalized
 * @param refractionRatio The ratio of the incoming to outgoing refractive indices
 */
inline vec3 refract(const vec3& v, const vec3& normal, float refractionRatio) {
    // Snell's law
    auto cosTheta = std::min(glm::dot(-v, normal), 1.0f);
    auto outgoingPerpendicular = refractionRatio * (v + cosTheta * normal);
    auto outgoingParallel = -std::sqrt(std::abs(1.0f - glm::length2(outgoingPerpendicular))) * normal;
    return outgoingPerpendicular + outgoingParallel;
};

/*
 * @param cosine The cosine of the angle between the incident ray and the normal
 * @param refractionRatio The ratio of the incoming to outgoing refractive indices
 */
inline float reflectance(float cosine, float refractionRatio) {
    // schlick's reflectance approximation
    auto r0 = (1 - refractionRatio) / (1 + refractionRatio);
    r0 *= r0;
    return r0 + (1 - r0) * static_cast<float>(std::pow((1 - cosine), 5));
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const glm::vec<2, T>& v) {
    os << "(" << v.x << ", " << v.y << ")";
    return os;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const glm::vec<3, T>& v) {
    os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return os;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const glm::vec<4, T>& v) {
    os << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
    return os;
}

// A closed interval [min, max]
template <typename T>
struct Interval {
    T min, max;

    constexpr Interval() = default;

    constexpr Interval(T min, T max) : min(min), max(max) {}

    constexpr T length() const { return max - min; }

    constexpr bool contains(T value) const { return value >= min && value <= max; }

    constexpr bool surrounds(T value) const { return value > min && value < max; }

    constexpr T clamp(T value) const { return std::max(min, std::min(max, value)); }

    static const Interval<T> empty;
    static const Interval<T> universe;
};

template <typename T>
const inline Interval<T> Interval<T>::empty = {std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest()};

template <typename T>
const inline Interval<T> Interval<T>::universe = {std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max()};
