#include "ProgressiveSolver.h"

void ProgressiveSolver::initialize(const Ref<const Scene>& scene) {
    IGISolver::initialize(scene);

    m_lightmap = Texture<vec3>(m_scene->lightmapSize());
    m_residues = Texture<vec3>(m_scene->lightmapSize());

    m_lightmap.clear(vec3(0));
    m_residues.clear(vec3(0));

    auto patchIdx = uvec2(0, 0);
    for (patchIdx.y = 0; patchIdx.y < m_lightmapSize.y; patchIdx.y++) {
        for (patchIdx.x = 0; patchIdx.x < m_lightmapSize.x; patchIdx.x++) {
            auto& patch = m_scene->patches()[patchIdx];
            if (patch.face == nullptr)
				continue;

            auto residue = patch.face->material->emission;
            m_residues[patchIdx] = residue;
            m_lightmap[patchIdx] = residue;

            if (glm::length2(residue) > glm::length2(m_residues[m_maxResiduePatchIdx]))
                m_maxResiduePatchIdx = patchIdx;
        }
    }
}

Texture<vec3> ProgressiveSolver::solve() {
    for (u32 i = 0;; i++) {
        auto residue = shoot(m_maxResiduePatchIdx);
        if (residue <= m_residueThreshold)
            break;

        //if (i % 100 == 0)
            LOG(std::format("{}: residue={:0.4f}", i, residue));
    }

    return std::move(m_lightmap);
}

f32 ProgressiveSolver::shoot(uvec2 sourceIdx) {
    auto& source = m_scene->patches()[sourceIdx];
    auto sourceResidue = m_residues[sourceIdx];
    auto shotRad = glm::length(sourceResidue);

    if (source.face == nullptr || shotRad <= m_residueThreshold)
        return 0;  // nothing to solve

    auto maxResiduePatch = uvec2(0, 0);
    f32 maxResidue2 = 0;  // squared magnitude

    f32 reflectedRad = 0;

    // shoot to other patches
    auto receiverIdx = uvec2(0, 0);
    for (receiverIdx.y = 0; receiverIdx.y < m_lightmapSize.y; receiverIdx.y++) {
        for (receiverIdx.x = 0; receiverIdx.x < m_lightmapSize.x; receiverIdx.x++) {
            auto& receiver = m_scene->patches()[receiverIdx];
            if (receiver.face == nullptr || receiver == source)
                continue;

            // check for max residue canditate before possible skipping
            auto residueMagnitude2 = glm::length2(m_residues[receiverIdx]);
            if (residueMagnitude2 > maxResidue2) {
                maxResidue2 = residueMagnitude2;
                maxResiduePatch = receiverIdx;
            }

            if (receiver.face == source.face)
                continue;

            // check if the patches are facing each other
            auto sightLine = glm::normalize(receiver.vertices[0] - source.vertices[0]);
            if (glm::dot(sightLine, source.face->normal) <= 0 || glm::dot(-sightLine, receiver.face->normal) <= 0)
                continue;

            auto F = calculateFormFactor(source, receiver, *m_scene);
            if (F == 0)
                continue;

            auto deltaRad = receiver.face->material->albedo * sourceResidue * F * source.area / receiver.area;
            m_residues[receiverIdx] += deltaRad;
            m_lightmap[receiverIdx] += deltaRad;
            reflectedRad += glm::length(deltaRad);

            // check for max residue canditate
            residueMagnitude2 = glm::length2(m_residues[receiverIdx]);
            if (residueMagnitude2 > maxResidue2) {
                maxResidue2 = residueMagnitude2;
                maxResiduePatch = receiverIdx;
            }
        }
    }

    m_residues[sourceIdx] = vec3(0);
    m_maxResiduePatchIdx = maxResiduePatch;

    // LOG(std::format("shotRad={:.04f} reflectedRadPer={:.02f}", shotRad, reflectedRad / shotRad));
    return shotRad;  // return the amount of light shot
}
