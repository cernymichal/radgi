#pragma once

#include <array>
#include <ostream>
#include <utility>

#include "Scalars.h"

#ifdef __CUDACC__

#define GLM_FORCE_CUDA
#define CUDA_VERSION 12400

#include <cuda_runtime.h>

#define MATH_CONSTEXPR
#define MATH_FUNC_QUALIFIER __device__ __host__ __forceinline__

#else

#define MATH_CONSTEXPR constexpr
#define MATH_FUNC_QUALIFIER __forceinline

#endif

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>

constexpr f64 E = 2.7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274;
constexpr f64 HALF_PI = 1.5707963267948966192313216916397514420985846996875529104874722961539082031431044993140174126710585339;
constexpr f64 PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679;
constexpr f64 TWO_PI = 6.2831853071795864769252867665590057683943387987502116419498891846156328125724179972560696506842341358;

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
using glm::u8vec3;
using glm::uvec2;
using glm::uvec3;
using glm::uvec4;
using glm::vec2;
using glm::vec3;
using glm::vec4;

// Comparison operators for vectors

template <int L, typename T, glm::qualifier Q>
MATH_CONSTEXPR MATH_FUNC_QUALIFIER glm::vec<L, bool> operator!(const glm::vec<L, T, Q>& v) {
    return glm::not_(v);
}

/*
template <int L, typename t, glm::qualifier Q>
MATH_CONSTEXPR MATH_FUNC_QUALIFIER glm::vec<L, bool> operator==(const glm::vec<L, t, Q>& a, const glm::vec<L, t, Q>& b) {
    return glm::equal(a, b);
}

template <int L, typename t, glm::qualifier Q>
MATH_CONSTEXPR MATH_FUNC_QUALIFIER glm::vec<L, bool> operator!=(const glm::vec<L, t, Q>& a, const glm::vec<L, t, Q>& b) {
    return glm::notEqual(a, b);
}
*/

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
// TODO use glm

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

namespace glm {

template <typename T, glm::qualifier Q>
MATH_CONSTEXPR MATH_FUNC_QUALIFIER T cross(const glm::vec<2, T, Q>& a, const glm::vec<2, T, Q>& b) {
    return a.x * b.y - a.y * b.x;
}

}  // namespace glm

/*
 * @param v The vector
 * @return The index of the dimension with the maximum value
 */
MATH_CONSTEXPR MATH_FUNC_QUALIFIER u8 maxDimension(const vec3& v) {
    if (v.x > v.y) {
        return v.x > v.z ? 0 : 2;
    }
    else {
        return v.y > v.z ? 1 : 2;
    }
}

/*
 * @param v The incoming vector
 * @param normal The normal vector, normalized
 */
MATH_CONSTEXPR MATH_FUNC_QUALIFIER vec3 reflect(const vec3& v, const vec3& normal) {
    return v - 2.0f * glm::dot(v, normal) * normal;
}

/*
 * @param v The incoming vector, normalized
 * @param normal The normal vector, normalized
 * @param refractionRatio The ratio of the incoming to outgoing refractive indices
 */
MATH_FUNC_QUALIFIER vec3 refract(const vec3& v, const vec3& normal, f32 refractionRatio) {
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
MATH_FUNC_QUALIFIER f32 reflectance(f32 cosine, f32 refractionRatio) {
    // Schlick's reflectance approximation
    auto r0 = (1 - refractionRatio) / (1 + refractionRatio);
    r0 *= r0;
    return r0 + (1 - r0) * static_cast<f32>(std::pow((1 - cosine), 5));
}

// a closed interval [min, max]
template <typename T>
struct Interval {
    T min, max;

    MATH_CONSTEXPR MATH_FUNC_QUALIFIER Interval() = default;

    MATH_CONSTEXPR MATH_FUNC_QUALIFIER Interval(T min, T max) : min(min), max(max) {}

    MATH_CONSTEXPR MATH_FUNC_QUALIFIER T length() const {
        return max - min;
    }

    MATH_CONSTEXPR MATH_FUNC_QUALIFIER T center() const {
        if constexpr (std::is_same_v<T, f32> || std::is_same_v<T, vec3>)
            return (min + max) / 2.0f;
        else
            return (min + max) / 2;
    }

    MATH_CONSTEXPR MATH_FUNC_QUALIFIER bool contains(T value) const {
        auto result = value >= min && value <= max;

        if constexpr (std::is_same_v<T, vec3>)  // TODO check for other vector types
            return glm::all(result);
        else
            return result;
    }

    MATH_CONSTEXPR MATH_FUNC_QUALIFIER bool contains(const Interval& other) const {
        auto result = other.min >= min && other.max <= max;

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

    MATH_CONSTEXPR MATH_FUNC_QUALIFIER bool surrounds(const Interval& other) const {
        auto result = other.min > min && other.max < max;

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

    MATH_CONSTEXPR MATH_FUNC_QUALIFIER Interval extendTo(const T& other) const {
        return {glm::min(min, other), glm::max(max, other)};
    }

    MATH_CONSTEXPR MATH_FUNC_QUALIFIER Interval pad(T padding) const {
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
    return {vec3(std::numeric_limits<f32>::max()), vec3(std::numeric_limits<f32>::lowest())};
}

template <>
MATH_CONSTEXPR MATH_FUNC_QUALIFIER static Interval<vec3> Interval<vec3>::universe() {
    return {vec3(std::numeric_limits<f32>::lowest()), vec3(std::numeric_limits<f32>::max())};
}

/*
 * @param aabb The AABB to calculate the surface area of
 * @return The surface area of the AABB
 */
MATH_CONSTEXPR MATH_FUNC_QUALIFIER f32 AABBSurfaceArea(const AABB& aabb) {
    auto size = aabb.max - aabb.min;
    return 2.0f * (size.x * size.y + size.x * size.z + size.y * size.z);
}

/*
 * @param aabb The AABB to calculate the volume of
 * @return The volume of the AABB
 */
MATH_CONSTEXPR MATH_FUNC_QUALIFIER f32 AABBvolume(const AABB& aabb) {
    auto size = aabb.max - aabb.min;
    return size.x * size.y * size.z;
}

/*
 * @param rayOrigin The origin of the ray
 * @param rayDirectionInv The inverse of the direction of the ray
 * @param box The AABB to check against
 * @return The tNear and tFar values of the intersection points along the ray, or NANs if there is no intersection
 */
MATH_CONSTEXPR MATH_FUNC_QUALIFIER Interval<f32> rayAABBintersection(const vec3& rayOrigin, const vec3& rayDirectionInv, const AABB& aabb) {
    // https://tavianator.com/2011/ray_box.html
    // https://gist.github.com/DomNomNom/46bb1ce47f68d255fd5d

    vec3 tMin = (aabb.min - rayOrigin) * rayDirectionInv;
    vec3 tMax = (aabb.max - rayOrigin) * rayDirectionInv;
    vec3 t1 = glm::min(tMin, tMax);
    vec3 t2 = glm::max(tMin, tMax);
    f32 tNear = std::max(std::max(t1.x, t1.y), t1.z);
    f32 tFar = std::min(std::min(t2.x, t2.y), t2.z);

    if (tNear > tFar)
        return {NAN, NAN};

    return {tNear, tFar};
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

    f32 t = glm::cross(bOrigin - aOrigin, bDirection / glm::cross(aDirection, bDirection));
    f32 u = glm::cross(aOrigin - bOrigin, aDirection / glm::cross(bDirection, aDirection));
    return vec2(t, u);
}

struct TriangleHit {
    f32 t;             // X = rayOrigin + rayDirection * t
    vec3 barycentric;  // X = vertices[0] * u + vertices[1] * v + vertices[2] * w
};

/*
 * Möller–Trumbore intersection algorithm
 *
 * @param rayOrigin The origin of the ray
 * @param rayDirection The direction of the ray
 * @param vertices The vertices of the triangle
 * @param intersectBackfacing Whether to intersect back facing triangles
 * @return The t from the ray origin along the rayDirection the intersection point, or NAN if there is no intersection, together with the barycentric coordinates
 */
MATH_CONSTEXPR MATH_FUNC_QUALIFIER TriangleHit rayTriangleIntersection(const vec3& rayOrigin, const vec3& rayDirection, const vec3& vertexA, const vec3& vertexB, const vec3& vertexC, bool backfaceCulling = true) {
    // https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm

    vec3 edge1 = vertexB - vertexA;
    vec3 edge2 = vertexC - vertexA;
    vec3 P = glm::cross(rayDirection, edge2);
    f32 determinant = glm::dot(edge1, P);

    // if the determinant is negative, the triangle is back facing
    // if the determinant is close to 0, the ray misses the triangle
    if ((backfaceCulling && determinant <= glm::epsilon<f32>()) ||
        (!backfaceCulling && std::abs(determinant) <= glm::epsilon<f32>()))
        return {NAN};

    f32 determinantInv = 1.0f / determinant;
    vec3 T = rayOrigin - vertexA;
    f32 v = glm::dot(T, P) * determinantInv;
    if (v < 0 || v > 1)
        return {NAN};

    vec3 Q = glm::cross(T, edge1);
    f32 w = glm::dot(rayDirection, Q) * determinantInv;
    if (w < 0 || v + w > 1)
        return {NAN};

    f32 t = glm::dot(edge2, Q) * determinantInv;
    if (t < 0)
        return {NAN};

    // X = rayOrigin + rayDirection * t
    // X = vertexA * u + vertexB * v + vertexC * w - Barycentric coordinates
    return {t, vec3((1 - v - w), v, w)};
}

/*
 * Ray shear constants for the Woop-Benthin-Wald intersection algorithm
 */
struct RayShearConstants {
    u8vec3 k;  // The indices of the dimensions, with k.z being the dimension where the ray direction is maximal
    vec3 s;    // The shear constants

    MATH_CONSTEXPR MATH_FUNC_QUALIFIER RayShearConstants(const vec3& rayDirection) {
        // https://jcgt.org/published/0002/01/05/paper.pdf

        // Calculate the dimension where the ray direction is maximal
        k.z = maxDimension(glm::abs(rayDirection));
        k.x = k.z == 2 ? 0 : k.z + 1;  // Maybe a faster modulo
        k.y = k.x == 2 ? 0 : k.x + 1;

        // Swap k.x and k.y dimension to preserve winding direction of triangles
        if (rayDirection[k.z] < 0.0f)
            std::swap(k.x, k.y);

        // Calculate shear constants
        s.z = 1.0f / rayDirection[k.z];
        s.x = rayDirection[k.x] * s.z;
        s.y = rayDirection[k.y] * s.z;
    }
};

/*
 * Woop-Benthin-Wald watertight intersection algorithm
 *
 * @param rayOrigin The origin of the ray
 * @param rayShearConstants The shear constants of the ray
 * @param vertexA The first vertex of the triangle
 * @param vertexB The second vertex of the triangle
 * @param vertexC The third vertex of the triangle
 * @param intersectBackfacing Whether to intersect back facing triangles
 * @return The t from the ray origin along the rayDirection the intersection point, or NAN if there is no intersection, together with the barycentric coordinates
 */
MATH_CONSTEXPR MATH_FUNC_QUALIFIER TriangleHit rayTriangleIntersectionWT(const vec3& rayOrigin, const RayShearConstants& rayShearConstants, const vec3& vertexA, const vec3& vertexB, const vec3& vertexC, bool backfaceCulling = true) {
    // https://jcgt.org/published/0002/01/05/paper.pdf

    // Calculate vertices relative to ray origin
    const vec3 a = vertexA - rayOrigin;
    const vec3 b = vertexB - rayOrigin;
    const vec3 c = vertexC - rayOrigin;

    // Perform shear and scale of vertices
    const u8vec3& k = rayShearConstants.k;
    const vec3& s = rayShearConstants.s;
    const f32 ax = a[k.x] - s.x * a[k.z];
    const f32 ay = a[k.y] - s.y * a[k.z];
    const f32 bx = b[k.x] - s.x * b[k.z];
    const f32 by = b[k.y] - s.y * b[k.z];
    const f32 cx = c[k.x] - s.x * c[k.z];
    const f32 cy = c[k.y] - s.y * c[k.z];

    // Calculate scaled barycentric coordinates of vertices
    f32 u = cx * by - cy * bx;
    f32 v = ax * cy - ay * cx;
    f32 w = bx * ay - by * ax;

    // Fallback to test against edges using f64 precision
    if (u == 0.0f || v == 0.0f || w == 0.0f) {
        f64 cxby = (f64)(cx) * (f64)(by);
        f64 cybx = (f64)(cy) * (f64)(bx);
        u = (f32)(cxby - cybx);
        f64 axcy = (f64)(ax) * (f64)(cy);
        f64 aycx = (f64)(ay) * (f64)(cx);
        v = (f32)(axcy - aycx);
        f64 bxay = (f64)(bx) * (f64)(ay);
        f64 byax = (f64)(by) * (f64)(ax);
        w = (f32)(bxay - byax);
    }

    // Perform edge tests
    // Moving this test before and at the end of the previous conditional gives higher performance
    if ((u < 0.0f || v < 0.0f || w < 0.0f) && (backfaceCulling || u > 0.0f || v > 0.0f || w > 0.0f))
        return {NAN};

    // Calculate determinant
    const f32 determinant = u + v + w;
    if (determinant == 0.0f)
        return {NAN};

    // Calculate scaled z-coordinates of vertices and use them to calculate the hit distance
    const f32 az = s.z * a[k.z];
    const f32 bz = s.z * b[k.z];
    const f32 cz = s.z * c[k.z];
    const f32 t = u * az + v * bz + w * cz;

    // Normalize t and barycentric coordinates
    const f32 determinantInv = 1.0f / determinant;
    return {t * determinantInv, vec3(u, v, w) * determinantInv};
}
