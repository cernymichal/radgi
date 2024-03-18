#pragma once

#include "IGISolver.h"

class GatheringSolver : public IGISolver {
public:
    GatheringSolver(uint32_t bounces = 4) : m_bounces(bounces) {
    }

    virtual void initialize(const Ref<const Scene>& scene) override;

    virtual Texture<vec3> solve() override;

protected:
    uint32_t m_bounces;
    Texture<vec3> m_lightmap;
    Texture<vec3> m_residues;
    Texture<vec3> m_nextResidues;

    float gather(uvec2 destinationIdx);
};
