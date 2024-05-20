#pragma once

#include <array>
#include <numbers>
#include <ostream>
#include <utility>

#include "Scalars.h"

#ifdef __CUDACC__

#include <cuda.h>
#include <cuda_runtime.h>

#define MATH_FUNC_QUALIFIER __device__ __host__

#define GLM_FORCE_CUDA

#else

#define MATH_FUNC_QUALIFIER

#endif

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>

using std::numbers::e_v;
using std::numbers::pi_v;

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
MATH_FUNC_QUALIFIER constexpr inline T cross(const glm::vec<2, T, Q>& a, const glm::vec<2, T, Q>& b) {
    return a.x * b.y - a.y * b.x;
}

}  // namespace glm

/*
 * @param v The incoming vector
 * @param normal The normal vector
 */
MATH_FUNC_QUALIFIER GLM_CONSTEXPR inline vec3 reflect(const vec3& v, const vec3& normal) {
    return v - 2.0f * glm::dot(v, normal) * normal;
}

/*
 * @param v The incoming vector, normalized
 * @param normal The normal vector, normalized
 * @param refractionRatio The ratio of the incoming to outgoing refractive indices
 */
MATH_FUNC_QUALIFIER inline vec3 refract(const vec3& v, const vec3& normal, f32 refractionRatio) {
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
MATH_FUNC_QUALIFIER GLM_CONSTEXPR inline vec2 lineIntersection(const vec2& aOrigin, const vec2& aDirection, const vec2& bOrigin, const vec2& bDirection) {
    // X = aOrigin + aDirection * t
    // X = bOrigin + bDirection * u

    f32 t = glm::cross(bOrigin - aOrigin, bDirection / glm::cross(aDirection, bDirection));  // TODO calc only one parameter
    f32 u = glm::cross(aOrigin - bOrigin, aDirection / glm::cross(bDirection, aDirection));
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
MATH_FUNC_QUALIFIER GLM_CONSTEXPR inline f32 rayTriangleIntersection(const vec3& rayOrigin, const vec3& rayDirection, const std::array<vec3, 3>& vertices) {
    // X = rayOrigin + rayDirection * t

    // Möller–Trumbore intersection algorithm
    auto edge1 = vertices[1] - vertices[0];
    auto edge2 = vertices[2] - vertices[0];
    auto P = glm::cross(rayDirection, edge2);
    auto determinant = glm::dot(edge1, P);

    // if the determinant is negative, the triangle is back facing
    // if the determinant is close to 0, the ray misses the triangle
    if (glm::abs(determinant) < 0.0001f)
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
MATH_FUNC_QUALIFIER inline f32 reflectance(f32 cosine, f32 refractionRatio) {
    // schlick's reflectance approximation
    auto r0 = (1 - refractionRatio) / (1 + refractionRatio);
    r0 *= r0;
    return r0 + (1 - r0) * static_cast<f32>(std::pow((1 - cosine), 5));
}

// Comparison operators for vectors

template <int L, typename T, glm::qualifier Q>
MATH_FUNC_QUALIFIER constexpr inline glm::vec<L, bool> operator!(const glm::vec<L, T, Q>& v) {
    return glm::not_(v);
}

template <int L, typename T, glm::qualifier Q>
MATH_FUNC_QUALIFIER constexpr inline glm::vec<L, bool> operator==(const glm::vec<L, T, Q>& a, const glm::vec<L, T, Q>& b) {
    return glm::equal(a, b);
}

template <int L, typename T, glm::qualifier Q>
MATH_FUNC_QUALIFIER constexpr inline glm::vec<L, bool> operator!=(const glm::vec<L, T, Q>& a, const glm::vec<L, T, Q>& b) {
    return glm::notEqual(a, b);
}

template <int L, typename T, glm::qualifier Q>
MATH_FUNC_QUALIFIER constexpr inline glm::vec<L, bool> operator>(const glm::vec<L, T, Q>& a, const glm::vec<L, T, Q>& b) {
    return glm::greaterThan(a, b);
}

template <int L, typename T, glm::qualifier Q>
MATH_FUNC_QUALIFIER constexpr inline glm::vec<L, bool> operator>=(const glm::vec<L, T, Q>& a, const glm::vec<L, T, Q>& b) {
    return glm::greaterThanEqual(a, b);
}

template <int L, typename T, glm::qualifier Q>
MATH_FUNC_QUALIFIER constexpr inline glm::vec<L, bool> operator<(const glm::vec<L, T, Q>& a, const glm::vec<L, T, Q>& b) {
    return glm::lessThan(a, b);
}

template <int L, typename T, glm::qualifier Q>
MATH_FUNC_QUALIFIER constexpr glm::vec<L, bool> operator<=(const glm::vec<L, T, Q>& a, const glm::vec<L, T, Q>& b) {
    return glm::lessThanEqual(a, b);
}

// Stream operators for vectors

template <typename T>
MATH_FUNC_QUALIFIER inline std::ostream& operator<<(std::ostream& os, const glm::vec<2, T>& v) {
    os << "(" << v.x << ", " << v.y << ")";
    return os;
}

template <typename T>
MATH_FUNC_QUALIFIER inline std::ostream& operator<<(std::ostream& os, const glm::vec<3, T>& v) {
    os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return os;
}

template <typename T>
MATH_FUNC_QUALIFIER inline std::ostream& operator<<(std::ostream& os, const glm::vec<4, T>& v) {
    os << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
    return os;
}

// A closed interval [min, max]
template <typename T>
struct Interval {
    T min, max;

    MATH_FUNC_QUALIFIER constexpr inline Interval() = default;

    MATH_FUNC_QUALIFIER constexpr inline Interval(T min, T max) : min(min), max(max) {}

    MATH_FUNC_QUALIFIER constexpr inline T length() const {
        return max - min;
    }

    MATH_FUNC_QUALIFIER constexpr inline T center() const {
        if constexpr (std::is_same_v<T, f32> || std::is_same_v<T, vec3>)
            return (min + max) / 2.0f;
        else
            return (min + max) / 2;
    }

    MATH_FUNC_QUALIFIER constexpr inline bool contains(T value) const {
        auto result = value >= min && value <= max;

        if constexpr (std::is_same_v<T, vec3>)
            return glm::all(result);
        else
            return result;
    }

    MATH_FUNC_QUALIFIER constexpr inline bool surrounds(T value) const {
        auto result = value > min && value < max;

        if constexpr (std::is_same_v<T, vec3>)
            return glm::all(result);
        else
            return result;
    }

    MATH_FUNC_QUALIFIER constexpr inline bool intersects(const Interval& other) const {
        auto result = min <= other.max && max >= other.min;

        if constexpr (std::is_same_v<T, vec3>)
            return glm::all(result);
        else
            return result;
    }

    MATH_FUNC_QUALIFIER constexpr inline Interval intersection(const Interval& other) const {
        return {glm::max(min, other.min), glm::min(max, other.max)};
    }

    MATH_FUNC_QUALIFIER constexpr inline Interval boundingUnion(const Interval& other) const {
        return {glm::min(min, other.min), glm::max(max, other.max)};
    }

    MATH_FUNC_QUALIFIER constexpr inline Interval expand(T padding) const {
        return Interval(min - padding, max + padding);
    }

    MATH_FUNC_QUALIFIER constexpr inline T clamp(T value) const {
        return glm::clamp(value, min, max);
    }

    MATH_FUNC_QUALIFIER constexpr static inline Interval<T> empty() {
        return {std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest()};
    }

    MATH_FUNC_QUALIFIER constexpr static inline Interval<T> universe() {
        return {std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max()};
    }
};

// Axis Aligned Bounding Box
using AABB = Interval<vec3>;

template <>
constexpr inline Interval<vec3> Interval<vec3>::empty() {
    return {vec3(std::numeric_limits<f32>::max()), vec3(std::numeric_limits<f32>::lowest())};
}

template <>
constexpr inline Interval<vec3> Interval<vec3>::universe() {
    return {vec3(std::numeric_limits<f32>::lowest()), vec3(std::numeric_limits<f32>::max())};
}

/*
 * @param rayOrigin The origin of the ray
 * @param rayDirectionInv The inverse of the direction of the ray
 * @param box The AABB to check against
 * @return The tNear and tFar values of the intersection points along the ray, or NANs if there is no intersection
 */
MATH_FUNC_QUALIFIER GLM_CONSTEXPR inline std::pair<f32, f32> rayAABBintersection(const vec3& rayOrigin, const vec3& rayDirectionInv, const AABB& box) {
    // https://tavianator.com/2011/ray_box.html
    // https://gist.github.com/DomNomNom/46bb1ce47f68d255fd5d

    vec3 t1 = (box.min - rayOrigin) * rayDirectionInv;
    vec3 t2 = (box.max - rayOrigin) * rayDirectionInv;
    vec3 tMin = glm::min(t1, t2);
    vec3 tMax = glm::max(t1, t2);
    f32 tNear = glm::max(tMin.x, glm::max(tMin.y, tMin.z));
    f32 tFar = glm::min(tMax.x, glm::min(tMax.y, tMax.z));

    if (tNear > tFar)
        return {NAN, NAN};

    return {tNear, tFar};
};

#undef MATH_FUNC_QUALIFIER
