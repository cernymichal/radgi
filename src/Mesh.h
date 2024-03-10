#pragma once

struct Material {
    vec3 albedo = vec3(0);
    vec3 emission = vec3(0);
};

struct Face {
    vec3 vertices[4];
    uint8_t vertexCount;
    vec3 normal;
    vec2 lightmapUVs[4];
    Ref<Material> material;

    bool operator==(const Face& other) const {
        return this == &other;
    }
};

std::vector<Face> loadMesh(const std::filesystem::path& filePath);

void saveMesh(const std::filesystem::path& filePath, const std::vector<Face>& faces);
