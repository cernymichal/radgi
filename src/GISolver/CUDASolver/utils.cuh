#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "CUDAStructs.h"

using namespace CUDAStructs;

#define checkCUDAError(ans) \
    { logCUDAError((ans), __FILE__, __LINE__); }

inline void logCUDAError(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess)
        fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
}

struct RNG {
    uint32_t state;

    __device__ explicit constexpr RNG(uint32_t seed) : state(seed) {}

    __device__ constexpr inline float operator()(float min = 0.0f, float max = 1.0f) {
        // https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
        // PCG PRNG
        state = state * 747796405u + 2891336453u;
        uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        auto randomValue = (word >> 22u) ^ word;

        // map to a float in [min, max]
        return min + (max - min) * static_cast<float>(randomValue) / static_cast<float>(uint32_t(-1));
    }
};

__device__ bool intersectsBVH(const Scene& scene, const vec3& rayOrigin, const vec3& rayDirection, const Interval<float>& tInterval, const uint32_t excludedFaces[2]) {
    auto rayDirectionInv = 1.0f / rayDirection;

    uint32_t stack[64];
    int32_t stackSize = 0;
    stack[stackSize++] = 0;

    while (stackSize > 0) {
        auto node = stack[--stackSize];
        
        if (node >= scene.bvh.nodeCount / 2) {
            // Leaf node

            auto faceId = scene.bvh.nodes[node].face;
            if (faceId == uint32_t(-1))
                continue;

            const auto& face = scene.faces[faceId];

            if (glm::dot(rayDirection, face.normal) >= 0)
                continue;

            auto t = rayTriangleIntersection(rayOrigin, rayDirection, face.vertices);

            if (!isnan(t) && tInterval.contains(t) && faceId != excludedFaces[0] && faceId != excludedFaces[1])
                return true;

            continue;
        }

        auto intersections = rayAABBintersection(rayOrigin, rayDirectionInv, scene.bvh.nodes[node].aabb);
        auto tNear = intersections.first;
        auto tFar = intersections.second;
        if (isnan(tNear) || tNear > tInterval.max || tFar < tInterval.min)
            continue;

        auto leftChild = 2 * node + 1;
        auto rightChild = 2 * node + 2;
        stack[stackSize++] = rightChild;
        stack[stackSize++] = leftChild;
    }

    return false;
}
