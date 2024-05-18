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

__device__ bool intersectsBVH(const Scene& scene, const vec3& rayOrigin, const vec3& rayDirection, const Interval<f32>& tInterval, const u32 excludedFaces[2]) {
    vec3 rayDirectionInv = 1.0f / rayDirection;

    u32 stack[64];
    i32 stackSize = 0;
    stack[stackSize++] = 0;

    while (stackSize > 0) {
        auto node = stack[--stackSize];
        auto faceId = scene.bvh.nodes[node].face;

        if (faceId != u32(-1)) {
            const auto& face = scene.faces[faceId];

            auto t = rayTriangleIntersection(rayOrigin, rayDirection, face.vertices);

            if (!isnan(t) && tInterval.contains(t) && faceId != excludedFaces[0] && faceId != excludedFaces[1])
                return true;

            continue;
        }

        if (node >= scene.bvh.nodeCount / 2)  // Leaf node
            continue;

        auto [tNear, tFar] = rayAABBintersection(rayOrigin, rayDirectionInv, scene.bvh.nodes[node].aabb);
        if (isnan(tNear) || tNear > tInterval.max || tFar < tInterval.min)
            continue;

        auto leftChild = 2 * node + 1;
        auto rightChild = 2 * node + 2;
        stack[stackSize++] = rightChild;
        stack[stackSize++] = leftChild;

// #define DEBUG_BVH_STACK
#ifdef DEBUG_BVH_STACK
        if (stackSize >= 64) {
            printf("BVH traversal stack overflow\n");
            return false;
        }
#endif
    }

    return false;
}
