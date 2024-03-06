#pragma once

#include "Mesh.h"
#include "Scene.h"
#include "Texture.h"

struct Patch {
    vec3 vertices[4];
    uint8_t vertexCount = 0;
    vec3 center = vec3(0);
    float area = 0;
    vec3 residue = vec3(0);
    Face* face = nullptr;

    Patch() = default;

    Patch(uint8_t vertexCount, vec3 vertices[], Face* face) : vertexCount(vertexCount), face(face) {
        memcpy(this->vertices, vertices, vertexCount * sizeof(vec3));

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

    void solveProgressive(float residueThreshold = 0.2);

    void solveUniform(uint32_t iterations = 4);

    const Texture<vec3>& lightmap() const {
        return m_lightmapAccumulated;
    }

    void addPadding(uint32_t radius = 2);

private:
    Ref<Scene> m_scene;
    uvec2 m_lightmapSize;
    Texture<Patch> m_lightmapPatches;
    Texture<vec3> m_lightmapAccumulated;

    uvec2 m_maxResiduePatchIdx = uvec2(0);

    float shoot(uvec2 source, float residueThreshold = 0);

    std::pair<float, float> calculateFormFactor(const Patch& source, const Patch& destination);
};
