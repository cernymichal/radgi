#pragma once

#include "Mesh.h"
#include "Scene.h"
#include "Texture.h"

struct Patch {
    std::array<vec3, 4> vertices;
    uint8_t vertexCount = 0;
    vec3 center = vec3(0);
    float area = 0;
    vec3 residue = vec3(0);
    Face* face = nullptr;

    Patch() = default;

    Patch(uint8_t vertexCount, const std::array<vec3, 4>& vertices, Face* face) : vertices(vertices), vertexCount(vertexCount), face(face) {
        center = vec3(0);
        for (uint8_t i = 0; i < vertexCount; i++)
            center += vertices[i];
        center /= vertexCount;

        // TODO - this is not correct for general quads
        area = glm::length(glm::cross(vertices[1] - vertices[0], vertices[2] - vertices[1]));
        if (vertexCount == 3)
            area /= 2;

        residue = face->material->emission;  //* area;
    }

    bool operator==(const Patch& other) const {
        return this == &other;
    }
};

class RadiositySolver {
public:
    RadiositySolver(uvec2 lightmapSize)
        : m_lightmapSize(lightmapSize), m_lightmapPatches(lightmapSize), m_lightmapAccumulated(lightmapSize) {
        m_lightmapPatches.clear(Patch());
        m_lightmapAccumulated.clear(vec3(0));
    }

    void initialize(const Ref<Scene>& scene);

    void solveShooting(float residueThreshold = 0.2);

    void solveGathering(uint32_t iterations = 4);

    const Texture<vec3>& lightmap() const {
        return m_lightmapAccumulated;
    }

    void dilateLightmap(uint32_t radius = 2);

    std::vector<Face> createPatchGeometry() const;

private:
    Ref<Scene> m_scene;
    uvec2 m_lightmapSize;
    Texture<Patch> m_lightmapPatches;
    Texture<vec3> m_lightmapAccumulated;

    uvec2 m_maxResiduePatchIdx = uvec2(0);

    float shoot(uvec2 sourceIdx, float residueThreshold = 0);

    float gather(uvec2 destinationIdx);

    float calculateFormFactor(const Patch& source, const Patch& destination);
};
