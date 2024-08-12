#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "CUDAStructs.h"

using namespace CUDAStructs;

#define checkCUDAError(ans) \
    { logCUDAError((ans), __FILE__, __LINE__); }

inline void logCUDAError(cudaError_t code, const char* file, i32 line) {
    if (code != cudaSuccess)
        fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
}

struct RNG {
    u32 state;

    __device__ explicit constexpr RNG(u32 seed) : state(seed) {}

    __device__ constexpr inline f32 operator()(f32 min = 0.0f, f32 max = 1.0f) {
        // https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
        // PCG PRNG
        state = state * 747796405u + 2891336453u;
        u32 word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        auto randomValue = (word >> 22u) ^ word;

        // map to a f32 in [min, max]
        return min + (max - min) * static_cast<f32>(randomValue) / static_cast<f32>(u32(-1));
    }
};

constexpr u32 BVH_MAX_DEPTH = 32;

__device__ bool intersectsBVH(const Scene& scene, const vec3& rayOrigin, const vec3& rayDirection, const Interval<f32>& tInterval, const u32 excludedFaces[2]) {
    vec3 rayDirectionInv = 1.0f / rayDirection;

    std::array<u32, BVH_MAX_DEPTH> stack;
    u32 stackSize = 0;
    stack[stackSize++] = 0;

    while (stackSize != 0) {
        u32 nodeIndex = stack[--stackSize];
        const BVH::Node& node = scene.bvh.nodes[nodeIndex];

        auto nodeIntersection = rayAABBintersection(rayOrigin, rayDirectionInv, node.aabb);
        if (isnan(nodeIntersection.min) || tInterval.intersection(nodeIntersection).length() < 0)
            continue;

        if (node.faceCount != 0) {
            // Leaf node
            for (u32 i = node.faceIndex; i < node.faceIndex + node.faceCount; i++) {
                if (i == excludedFaces[0] || i == excludedFaces[1])
                    continue;

                const Face& face = scene.faces[i];
                auto [t, barycentric] = rayTriangleIntersection(rayOrigin, rayDirection, face.vertices[0], face.vertices[1], face.vertices[2], false);
                if (!isnan(t) && tInterval.surrounds(t))
                    return true;
            }

            continue;
        }

        // Add children to stack
        stack[stackSize++] = node.childIndex;
        stack[stackSize++] = node.childIndex + 1;

        // #define DEBUG_BVH_STACK
#ifdef DEBUG_BVH_STACK
        if (stackSize >= BVH_MAX_DEPTH) {
            printf("BVH traversal stack overflow\n");
            return false;
        }
#endif
    }

    return false;
}
