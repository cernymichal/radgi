#pragma once

struct Material {
    vec3 albedo = vec3(0);
    vec3 emission = vec3(0);
};

struct Face {
    std::array<vec3, 3> vertices; // CCW, only triangles are supported
    vec3 normal;
    std::array<vec2, 3> lightmapUVs;
    Ref<Material> material;

    bool operator==(const Face& other) const {
        return this == &other;
    }
};

std::vector<Face> loadMesh(const std::filesystem::path& filePath);

void saveMesh(const std::filesystem::path& filePath, const std::vector<Face>& faces);
