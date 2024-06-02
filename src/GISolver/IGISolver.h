#pragma once

#include "../Scene.h"

class IGISolver {
public:
    virtual ~IGISolver() = default;

    virtual void initialize(const Ref<const Scene>& scene) {
        m_scene = scene;
        m_lightmapSize = m_scene->lightmapSize();
    }

    virtual Texture<vec3> solve() = 0;

protected:
    Ref<const Scene> m_scene = nullptr;
    uvec2 m_lightmapSize = uvec2(0);

    IGISolver() = default;
};

f32 calculateFormFactor(const Patch& patchA, const Patch& patchB, const Scene& scene);
