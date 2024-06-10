#pragma once

#include "Utils/Log.h"
#include "Utils/Math.h"
#include "Utils/Scalars.h"
#include "configuration.h"

#ifdef USE_FP16

#ifdef __CUDACC__

#include <cuda_fp16.h>
typedef half f16;

#else

#include <fp16.h>
typedef u16 f16;

#endif  // __CUDACC__

#else

typedef f32 f16;

#endif  // USE_FP16

typedef glm::vec<3, f16> hvec3;

namespace CUDAStructs {

constexpr u32 NULL_ID = u32(-1);

struct Material {
    hvec3 albedo;
    hvec3 emission;
};

struct Face {
    std::array<vec3, 3> vertices;
    vec3 normal;
    u16 materialId;  // TODO can this indirection be removed?

    bool operator==(const Face& other) const {
        return this == &other;
    }
};

struct Patch {
    std::array<vec3, 4> vertices;
    u8 vertexCount;
    f16 area;
    u32 faceId;

    bool operator==(const Patch& other) const {
        return this == &other;
    }
};

struct BVH {
    struct Node {
        AABB aabb = AABB::empty();
        u32 face = u32(-1);
    };

    Node* nodes;
    u32 nodeCount;
};

struct Scene {
    Patch* patches;
    Face* faces;
    u32 faceCount;
    Material* materials;
    u32 materialCount;
    BVH bvh;
};

}  // namespace CUDAStructs
