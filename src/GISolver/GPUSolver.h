#pragma once

#include "IGISolver.h"

class GPUSolver : public IGISolver {
public:
    GPUSolver(uint32_t bounces = 4) : m_bounces(bounces) {
    }

    virtual void initialize(const Ref<const Scene>& scene) override;

    virtual Texture<vec3> solve() override;

protected:
    uint32_t m_bounces;
};
