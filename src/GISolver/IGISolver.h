#pragma once

#include "../Scene.h"

class IGISolver {
public:
    virtual ~IGISolver() = default;

    virtual void initialize(const Ref<const Scene>& scene) {
        m_scene = scene;
        m_lightmapSize = m_scene->lightmapSize();
    }

    virtual Texture<vec3> solve() abstract;

protected:
    Ref<const Scene> m_scene = nullptr;
    uvec2 m_lightmapSize;

    IGISolver() = default;
};

float calculateFormFactor(const Patch& patchA, const Patch& patchB, const Scene& scene);
