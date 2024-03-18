#pragma once

#include "IGISolver.h"

class ProgressiveSolver : public IGISolver {
public:
    ProgressiveSolver(float residueThreshold = 0.2) : m_residueThreshold(residueThreshold) {
    }

    virtual void initialize(const Ref<const Scene>& scene) override;

    virtual Texture<vec3> solve() override;

protected:
    float m_residueThreshold;
    Texture<vec3> m_lightmap;
    Texture<vec3> m_residues;

    uvec2 m_maxResiduePatchIdx = uvec2(0);

    float shoot(uvec2 sourceIdx);
};
