#pragma once

#include "BVH.h"
#include "Mesh.h"
#include "Texture.h"

struct Patch {
    std::array<vec3, 4> vertices;
    u8 vertexCount = 0;
    f32 area = 0;
    Face* face = nullptr;

    Patch() = default;

    Patch(u8 vertexCount, const std::array<vec3, 4>& vertices, Face* face) : vertices(vertices), vertexCount(vertexCount), face(face) {
        // TODO - this is not correct for general quads - https://en.wikipedia.org/wiki/Shoelace_formula
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
    Scene() : m_bvh(m_faces) {}

    void addMesh(const std::vector<Face>& mesh) {
        m_faces.reserve(m_faces.size() + mesh.size());
        for (const auto& face : mesh) {
            m_faces.push_back(face);
            m_materials.insert(face.material);
        }
    }

    void initialize(const uvec2& lightmapSize);

    const std::vector<Face>& faces() const {
        return m_faces;
    }

    const BVH& bvh() const {
        return m_bvh;
    }

    const std::unordered_set<Ref<Material>>& materials() const {
        return m_materials;
    }

    const uvec2& lightmapSize() const {
        return m_lightmapSize;
    }

    const Texture<Patch>& patches() const {
        return m_patches;
    }

    std::vector<Face> createPatchGeometry() const;

    void dilateLightmap(Texture<vec3>& lightmap, u32 radius = 2);  // TODO move out of here and make const

private:
    std::vector<Face> m_faces;
    BVH m_bvh;

    std::unordered_set<Ref<Material>> m_materials;
    uvec2 m_lightmapSize;
    Texture<Patch> m_patches;

    void createPatches();
};
