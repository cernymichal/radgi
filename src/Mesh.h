#pragma once

struct Material {
    vec3 albedo = vec3(0);
    vec3 emission = vec3(0);
};

struct Face {
    std::array<vec3, 4> vertices;
    uint8_t vertexCount;
    vec3 normal;
    std::array<vec2, 4> lightmapUVs;
    Ref<Material> material;

    bool operator==(const Face& other) const {
        return this == &other;
    }
};

std::vector<Face> loadMesh(const std::filesystem::path& filePath);

void saveMesh(const std::filesystem::path& filePath, const std::vector<Face>& faces);

template <int C, int L, typename T, glm::qualifier Q>
constexpr inline const std::array<glm::vec<L, T, Q>, 3>& narrowToTriangle(const std::array<glm::vec<L, T, Q>, C>& array) {
    return reinterpret_cast<const std::array<glm::vec<L, T, Q>, 3>&>(array);
}
