#pragma once

#include "Mesh.h"
#include "Texture.h"

struct Patch {
    std::array<vec3, 4> vertices;
    uint8_t vertexCount = 0;
    vec3 center = vec3(0);
    float area = 0;
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
    }

    bool operator==(const Patch& other) const {
        return this == &other;
    }
};

class Scene {
public:
    Scene() = default;

    void addMesh(const std::vector<Face>& mesh) {
        m_faces.insert(m_faces.end(), mesh.begin(), mesh.end());
    }

    void initialize(const uvec2& lightmapSize);

    const std::vector<Face>& faces() const {
        return m_faces;
    }

    const uvec2& lightmapSize() const {
        return m_lightmapSize;
    }

    const Texture<Patch>& patches() const {
        return m_patches;
    }

    std::vector<Face> createPatchGeometry() const;

    void dilateLightmap(Texture<vec3>& lightmap, uint32_t radius = 2); // TODO move out of here and make const

private:
    std::vector<Face> m_faces;
    uvec2 m_lightmapSize;
    Texture<Patch> m_patches;
};
