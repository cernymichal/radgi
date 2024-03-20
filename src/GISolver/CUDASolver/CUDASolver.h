#pragma once

#include "../IGISolver.h"
#include "CUDAStructs.h"

class CUDASolver : public IGISolver {
public:
    CUDASolver(uint32_t bounces = 4) : m_bounces(bounces) {
    }

    virtual void initialize(const Ref<const Scene>& scene) override;

    virtual Texture<vec3> solve() override;

protected:
    uint32_t m_bounces;
    int2 m_lightmapSize;
    std::vector<CUDAStructs::Material> m_materials;
    std::vector<CUDAStructs::Face> m_faces;
    std::vector<CUDAStructs::Patch> m_patches;
};
