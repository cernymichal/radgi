#pragma once

#ifdef __CUDACC__

#define GLM_FORCE_CUDA
#define CUDA_VERSION 12400

#include <cuda_runtime.h>

#define MATH_CONSTEXPR
#define MATH_FUNC_QUALIFIER __device__ __host__ __forceinline__

#include <cuda/std/array>
#define MATH_ARRAY cuda::std::array

#else

#define MATH_CONSTEXPR constexpr
#define MATH_FUNC_QUALIFIER inline
#define MATH_ARRAY std::array

#endif

#define GLM_ENABLE_EXPERIMENTAL

#include <array>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <ostream>
#include <utility>

MATH_ARRAY<float, 3> toStdArray(const glm::vec3& v) {
    return {v.x, v.y, v.z};
}

constexpr double PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679;
constexpr double TWO_PI = 6.2831853071795864769252867665590057683943387987502116419498891846156328125724179972560696506842341358;
constexpr double HALF_PI = 1.5707963267948966192313216916397514420985846996875529104874722961539082031431044993140174126710585339;

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

namespace glm {

template <typename T, glm::qualifier Q>
MATH_CONSTEXPR MATH_FUNC_QUALIFIER T cross(const glm::vec<2, T, Q>& a, const glm::vec<2, T, Q>& b) {
    return a.x * b.y - a.y * b.x;
}

}  // namespace glm

/*
 * @param v The incoming vector
 * @param normal The normal vector
 */
MATH_CONSTEXPR MATH_FUNC_QUALIFIER vec3 reflect(const vec3& v, const vec3& normal) {
    return v - 2.0f * glm::dot(v, normal) * normal;
}

/*
 * @param v The incoming vector, normalized
 * @param normal The normal vector, normalized
 * @param refractionRatio The ratio of the incoming to outgoing refractive indices
 */
MATH_FUNC_QUALIFIER vec3 refract(const vec3& v, const vec3& normal, float refractionRatio) {
    // Snell's law
    auto cosTheta = std::min(glm::dot(-v, normal), 1.0f);
    auto outgoingPerpendicular = refractionRatio * (v + cosTheta * normal);
    auto outgoingParallel = -std::sqrt(std::abs(1.0f - glm::length2(outgoingPerpendicular))) * normal;
    return outgoingPerpendicular + outgoingParallel;
};

/*
 * @param aOrigin The origin of the first line
 * @param aDirection The direction of the first line
 * @param bOrigin The origin of the second line
 * @param bDirection The direction of the second line
 * @return The parameters t and u from the line origins along the line directions to the intersection point
 *
 * @note If the lines are parallel, the result is undefined
 */
MATH_CONSTEXPR MATH_FUNC_QUALIFIER vec2 lineIntersection(const vec2& aOrigin, const vec2& aDirection, const vec2& bOrigin, const vec2& bDirection) {
    // X = aOrigin + aDirection * t
    // X = bOrigin + bDirection * u

    float t = glm::cross(bOrigin - aOrigin, bDirection / glm::cross(aDirection, bDirection));
    float u = glm::cross(aOrigin - bOrigin, aDirection / glm::cross(bDirection, aDirection));
    return vec2(t, u);
}

/*
 * @param rayOrigin The origin of the ray
 * @param rayDirection The direction of the ray
 * @param vertices The vertices of the triangle
 * @return The t from the ray origin along the rayDirection the intersection point, or NAN if there is no intersection
 *
 * @note back facing triangles are not intersected
 */
MATH_CONSTEXPR MATH_FUNC_QUALIFIER float rayTriangleIntersection(const vec3& rayOrigin, const vec3& rayDirection, const MATH_ARRAY<vec3, 3>& vertices) {
    // X = rayOrigin + rayDirection * t

    // Möller–Trumbore intersection algorithm
    auto edge1 = vertices[1] - vertices[0];
    auto edge2 = vertices[2] - vertices[0];
    auto P = glm::cross(rayDirection, edge2);
    auto determinant = glm::dot(edge1, P);

    // if the determinant is negative, the triangle is back facing
    // if the determinant is close to 0, the ray misses the triangle
    if (determinant < 0.0001f)
        return NAN;

    auto determinantInv = 1.0f / determinant;
    auto T = rayOrigin - vertices[0];
    auto u = glm::dot(T, P) * determinantInv;
    if (u < 0 || u > 1)
        return NAN;

    auto Q = glm::cross(T, edge1);
    auto v = glm::dot(rayDirection, Q) * determinantInv;
    if (v < 0 || u + v > 1)
        return NAN;

    auto t = glm::dot(edge2, Q) * determinantInv;
    if (t < 0)
        return NAN;

    return t;
}

/*
 * @param cosine The cosine of the angle between the incident ray and the normal
 * @param refractionRatio The ratio of the incoming to outgoing refractive indices
 */
MATH_FUNC_QUALIFIER float reflectance(float cosine, float refractionRatio) {
    // schlick's reflectance approximation
    auto r0 = (1 - refractionRatio) / (1 + refractionRatio);
    r0 *= r0;
    return r0 + (1 - r0) * static_cast<float>(std::pow((1 - cosine), 5));
}

// Comparison operators for vectors

template <int L, typename T, glm::qualifier Q>
MATH_CONSTEXPR MATH_FUNC_QUALIFIER glm::vec<L, bool> operator!(const glm::vec<L, T, Q>& v) {
    return glm::not_(v);
}

template <int L, typename T, glm::qualifier Q>
MATH_CONSTEXPR MATH_FUNC_QUALIFIER glm::vec<L, bool> operator==(const glm::vec<L, T, Q>& a, const glm::vec<L, T, Q>& b) {
    return glm::equal(a, b);
}

template <int L, typename T, glm::qualifier Q>
MATH_CONSTEXPR MATH_FUNC_QUALIFIER glm::vec<L, bool> operator!=(const glm::vec<L, T, Q>& a, const glm::vec<L, T, Q>& b) {
    return glm::notEqual(a, b);
}

template <int L, typename T, glm::qualifier Q>
MATH_CONSTEXPR MATH_FUNC_QUALIFIER glm::vec<L, bool> operator>(const glm::vec<L, T, Q>& a, const glm::vec<L, T, Q>& b) {
    return glm::greaterThan(a, b);
}

template <int L, typename T, glm::qualifier Q>
MATH_CONSTEXPR MATH_FUNC_QUALIFIER glm::vec<L, bool> operator>=(const glm::vec<L, T, Q>& a, const glm::vec<L, T, Q>& b) {
    return glm::greaterThanEqual(a, b);
}

template <int L, typename T, glm::qualifier Q>
MATH_CONSTEXPR MATH_FUNC_QUALIFIER glm::vec<L, bool> operator<(const glm::vec<L, T, Q>& a, const glm::vec<L, T, Q>& b) {
    return glm::lessThan(a, b);
}

template <int L, typename T, glm::qualifier Q>
MATH_CONSTEXPR MATH_FUNC_QUALIFIER glm::vec<L, bool> operator<=(const glm::vec<L, T, Q>& a, const glm::vec<L, T, Q>& b) {
    return glm::lessThanEqual(a, b);
}

// Stream operators for vectors

template <typename T>
MATH_FUNC_QUALIFIER std::ostream& operator<<(std::ostream& os, const glm::vec<2, T>& v) {
    os << "(" << v.x << ", " << v.y << ")";
    return os;
}

template <typename T>
MATH_FUNC_QUALIFIER std::ostream& operator<<(std::ostream& os, const glm::vec<3, T>& v) {
    os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return os;
}

template <typename T>
MATH_FUNC_QUALIFIER std::ostream& operator<<(std::ostream& os, const glm::vec<4, T>& v) {
    os << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
    return os;
}

// A closed interval [min, max]
template <typename T>
struct Interval {
    T min, max;

    MATH_CONSTEXPR MATH_FUNC_QUALIFIER Interval() = default;

    MATH_CONSTEXPR MATH_FUNC_QUALIFIER Interval(T min, T max) : min(min), max(max) {}

    MATH_CONSTEXPR MATH_FUNC_QUALIFIER T length() const {
        return max - min;
    }

    MATH_CONSTEXPR MATH_FUNC_QUALIFIER T center() const {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, vec3>)
            return (min + max) / 2.0f;
        else
            return (min + max) / 2;
    }

    MATH_CONSTEXPR MATH_FUNC_QUALIFIER bool contains(T value) const {
        auto result = value >= min && value <= max;

        if constexpr (std::is_same_v<T, vec3>)
            return glm::all(result);
        else
            return result;
    }

    MATH_CONSTEXPR MATH_FUNC_QUALIFIER bool surrounds(T value) const {
        auto result = value > min && value < max;

        if constexpr (std::is_same_v<T, vec3>)
            return glm::all(result);
        else
            return result;
    }

    MATH_CONSTEXPR MATH_FUNC_QUALIFIER bool intersects(const Interval& other) const {
        auto result = min <= other.max && max >= other.min;

        if constexpr (std::is_same_v<T, vec3>)
            return glm::all(result);
        else
            return result;
    }

    MATH_CONSTEXPR MATH_FUNC_QUALIFIER Interval intersection(const Interval& other) const {
        return {glm::max(min, other.min), glm::min(max, other.max)};
    }

    MATH_CONSTEXPR MATH_FUNC_QUALIFIER Interval boundingUnion(const Interval& other) const {
        return {glm::min(min, other.min), glm::max(max, other.max)};
    }

    MATH_CONSTEXPR MATH_FUNC_QUALIFIER Interval expand(T padding) const {
        return Interval(min - padding, max + padding);
    }

    MATH_CONSTEXPR MATH_FUNC_QUALIFIER T clamp(T value) const {
        return glm::clamp(value, min, max);
    }

    MATH_CONSTEXPR MATH_FUNC_QUALIFIER static Interval<T> empty() {
        return {std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest()};
    }

    MATH_CONSTEXPR MATH_FUNC_QUALIFIER static Interval<T> universe() {
        return {std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max()};
    }
};

// Axis Aligned Bounding Box
using AABB = Interval<vec3>;

template <>
MATH_CONSTEXPR MATH_FUNC_QUALIFIER static Interval<vec3> Interval<vec3>::empty() {
    return {vec3(std::numeric_limits<float>::max()), vec3(std::numeric_limits<float>::lowest())};
}

template <>
MATH_CONSTEXPR MATH_FUNC_QUALIFIER static Interval<vec3> Interval<vec3>::universe() {
    return {vec3(std::numeric_limits<float>::lowest()), vec3(std::numeric_limits<float>::max())};
}

/*
 * @param rayOrigin The origin of the ray
 * @param rayDirectionInv The inverse of the direction of the ray
 * @param box The AABB to check against
 * @return The tNear and tFar values of the intersection points along the ray, or NANs if there is no intersection
 */
MATH_CONSTEXPR MATH_FUNC_QUALIFIER std::pair<float, float> rayAABBintersection(const vec3& rayOrigin, const vec3& rayDirectionInv, const AABB& box) {
    // https://tavianator.com/2011/ray_box.html
    // https://gist.github.com/DomNomNom/46bb1ce47f68d255fd5d

    vec3 tMin = (box.min - rayOrigin) * rayDirectionInv;
    vec3 tMax = (box.max - rayOrigin) * rayDirectionInv;
    vec3 t1 = glm::min(tMin, tMax);
    vec3 t2 = glm::max(tMin, tMax);
    float tNear = std::max(std::max(t1.x, t1.y), t1.z);
    float tFar = std::min(std::min(t2.x, t2.y), t2.z);

    if (tNear > tFar)
        return {NAN, NAN};

    return {tNear, tFar};
};
