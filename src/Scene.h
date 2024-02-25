#pragma once

#include "Mesh.h"
#include "Texture.h"

struct Patch {
    vec3 position = vec3(0);
    vec2 size = vec2(0);  // are you sure?
    vec3 residue = vec3(0);
    Face* face = nullptr;
};

struct Scene {
    std::vector<Face> faces;
    Texture<Patch> lightmapPatches;
    Texture<vec3> lightmapAccumulated;

    Scene(uvec2 lightmapSize) : lightmapPatches(lightmapSize), lightmapAccumulated(lightmapSize) {
        lightmapPatches.clear(Patch());
        lightmapAccumulated.clear(vec3(0));
    }
};
