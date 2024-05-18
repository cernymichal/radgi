#pragma once

#include "Utils/Log.h"
#include "Utils/Math.h"
#include "Utils/Scalars.h"

namespace CUDAStructs {

constexpr u32 NULL_ID = u32(-1);

struct Material {
    vec3 albedo;
    vec3 emission;
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
    f32 area;
    u32 faceId;

    bool operator==(const Patch& other) const {
        return this == &other;
    }
};

struct BVH {
    struct Node {
        AABB aabb;
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
