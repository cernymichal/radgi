#pragma once

#include "IGISolver.h"

class ProgressiveSolver : public IGISolver {
public:
    ProgressiveSolver(f32 residueThreshold = 0.2) : m_residueThreshold(residueThreshold) {
    }

    virtual void initialize(const Ref<const Scene>& scene) override;

    virtual Texture<vec3> solve() override;

protected:
    f32 m_residueThreshold;
    Texture<vec3> m_lightmap;
    Texture<vec3> m_residues;

    uvec2 m_maxResiduePatchIdx = uvec2(0);

    f32 shoot(uvec2 sourceIdx);
};
