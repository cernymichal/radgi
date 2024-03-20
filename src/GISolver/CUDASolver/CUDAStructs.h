#pragma once

#include <vector_types.h>  // CUDA vector types

namespace CUDAStructs {

constexpr uint32_t NULL_ID = static_cast<uint32_t>(-1);

struct Material {
    float3 albedo;
    float3 emission;
};

struct Face {
    float3 vertices[3];
    float3 normal;
    uint16_t materialId;  // TODO can this indirection be removed?

    bool operator==(const Face& other) const {
        return this == &other;
    }
};

struct Patch {
    float3 vertices[4];
    uint8_t vertexCount;
    float3 center;  // TODO is this nessesary?
    float area;
    uint32_t faceId;

    bool operator==(const Patch& other) const {
        return this == &other;
    }
};

}  // namespace CUDAStructs
