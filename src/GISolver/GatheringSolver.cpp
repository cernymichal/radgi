#include "GatheringSolver.h"

void GatheringSolver::initialize(const Ref<const Scene>& scene) {
    IGISolver::initialize(scene);

    m_lightmap = Texture<vec3>(m_scene->lightmapSize());
    m_residues = Texture<vec3>(m_scene->lightmapSize());
    m_nextResidues = Texture<vec3>(m_scene->lightmapSize());

    m_lightmap.clear(vec3(0));
    m_residues.clear(vec3(0));
    m_nextResidues.clear(vec3(0));

    auto patchIdx = uvec2(0, 0);
    for (patchIdx.y = 0; patchIdx.y < m_lightmapSize.y; patchIdx.y++) {
        for (patchIdx.x = 0; patchIdx.x < m_lightmapSize.x; patchIdx.x++) {
            auto& patch = m_scene->patches()[patchIdx];
            if (patch.face == nullptr)
                continue;

            auto residue = patch.face->material->emission;
            m_residues[patchIdx] = residue;
            m_lightmap[patchIdx] = residue;
        }
    }
}

Texture<vec3> GatheringSolver::solve() {
    for (uint32_t i = 0; i < m_bounces; i++) {
#pragma omp parallel for
        for (uint32_t y = 0; y < m_lightmapSize.y; y++) {
            auto destinationIdx = uvec2(0, y);
            for (destinationIdx.x = 0; destinationIdx.x < m_lightmapSize.x; destinationIdx.x++) {
                gather(destinationIdx);
            }
        }

        m_residues = std::move(m_nextResidues);
        m_nextResidues = Texture<vec3>(m_lightmapSize);
        m_nextResidues.clear(vec3(0));

        LOG(std::format("Bounce {}/{}", i + 1, m_bounces));
    }

    return std::move(m_lightmap);
}

float GatheringSolver::gather(uvec2 destinationIdx) {
    auto& destination = m_scene->patches()[destinationIdx];

    if (destination.face == nullptr)
        return 0;  // nothing to solve

    float gatheredRad = 0;

    // shoot to other patches
    auto shooterIdx = uvec2(0, 0);
    for (shooterIdx.y = 0; shooterIdx.y < m_lightmapSize.y; shooterIdx.y++) {
        for (shooterIdx.x = 0; shooterIdx.x < m_lightmapSize.x; shooterIdx.x++) {
            auto& shooter = m_scene->patches()[shooterIdx];
            auto shooterResidue = m_residues[shooterIdx];
            if (shooter.face == nullptr || shooter == destination || shooter.face == destination.face || shooterResidue == vec3(0))
                continue;

            // check if the patches are facing each other
            auto sightLine = glm::normalize(shooter.vertices[0] - destination.vertices[0]);
            if (glm::dot(sightLine, destination.face->normal) <= 0 || glm::dot(-sightLine, shooter.face->normal) <= 0)
                continue;

            auto F = calculateFormFactor(shooter, destination, *m_scene);
            if (F == 0)
                continue;

            auto deltaRad = destination.face->material->albedo * shooterResidue * F * shooter.area / destination.area;
            m_nextResidues[destinationIdx] += deltaRad;
            m_lightmap[destinationIdx] += deltaRad;
        }
    }

    return gatheredRad;  // return the amount of light gathered
}
