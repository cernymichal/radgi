#pragma once

#include "Texture.h"

struct Material {
    vec3 albedo = vec3(0);
    vec3 emission = vec3(0);
};

struct Face {
    vec3 vertices[3];  // world space
    vec3 normal;
    vec2 lightmapUVs[3];
    Ref<Material> material;
};

std::vector<Face> loadMesh(const std::string& filePath);
