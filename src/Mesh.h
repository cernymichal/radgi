#pragma once

struct Material {
    vec3 albedo = vec3(0);
    vec3 emission = vec3(0);
};

struct Face {
    std::array<vec3, 3> vertices;  // CCW, only triangles are supported
    vec3 normal;
    std::array<vec2, 3> lightmapUVs;
    Ref<Material> material;
    AABB aabb;  // only used for BVH construction

    bool operator==(const Face& other) const {
        return this == &other;
    }

    AABB calculateAABB() const {
        AABB aabb = {vec3(std::numeric_limits<float>::max()), vec3(std::numeric_limits<float>::lowest())};
        for (const vec3& vertex : vertices) {
            aabb.min = min(aabb.min, vertex);
            aabb.max = max(aabb.max, vertex);
        }
        return aabb;
    }
};

std::vector<Face> loadMesh(const std::filesystem::path& filePath);

void saveMesh(const std::filesystem::path& filePath, const std::vector<Face>& faces);
