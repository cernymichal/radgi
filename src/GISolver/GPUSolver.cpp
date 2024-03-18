#include "GPUSolver.h"

void GPUSolver::initialize(const Ref<const Scene>& scene) {
    IGISolver::initialize(scene);
}

Texture<vec3> GPUSolver::solve() {
    return Texture<vec3>(m_lightmapSize);
}
