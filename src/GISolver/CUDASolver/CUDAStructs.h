#pragma once

#include "Utils/Log.h"
#include "Utils/Math.h"

namespace CUDAStructs {

constexpr uint32_t NULL_ID = uint32_t(-1);

struct Material {
    vec3 albedo;
    vec3 emission;
};

struct Face {
    MATH_ARRAY<vec3, 3> vertices;
    vec3 normal;
    uint16_t materialId;  // TODO can this indirection be removed?

    bool operator==(const Face& other) const {
        return this == &other;
    }
};

struct Patch {
    MATH_ARRAY<vec3, 4> vertices;
    uint8_t vertexCount;
    float area;
    uint32_t faceId;

    bool operator==(const Patch& other) const {
        return this == &other;
    }
};

struct BVH {
    struct Node {
        AABB aabb;
        uint32_t face = uint32_t(-1);
    };

    Node* nodes;
    uint32_t nodeCount;
};

struct Scene {
    Patch* patches;
    Face* faces;
    uint32_t faceCount;
    Material* materials;
    uint32_t materialCount;
    BVH bvh;
};

}  // namespace CUDAStructs
