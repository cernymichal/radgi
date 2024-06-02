#pragma once

#include "../IGISolver.h"
#include "CUDAStructs.h"

class CUDASolver : public IGISolver {
public:
    CUDASolver(u32 bounces = 4) : m_bounces(bounces) {
    }

    virtual void initialize(const Ref<const Scene>& scene) override;

    virtual Texture<vec3> solve() override;

protected:
    u32 m_bounces;
    uvec2 m_lightmapSize = uvec2(0);
    std::vector<CUDAStructs::Material> m_materials;
    std::vector<CUDAStructs::Face> m_faces;
    std::vector<CUDAStructs::Patch> m_patches;
    std::vector<CUDAStructs::BVH::Node> m_bvhNodes;
};
