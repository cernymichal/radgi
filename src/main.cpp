#include "Scene.h"

float triangleArea(const vec2& a, const vec2& b, const vec2& c) {
    return 0.5f * ((c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x));
}

vec3 barycentricCoordinates(const vec2 vertices[3], const vec2& point) {
    vec3 barycentric;
    barycentric.x = triangleArea(vertices[1], vertices[2], point);
    barycentric.y = triangleArea(vertices[2], vertices[0], point);
    barycentric.z = triangleArea(vertices[0], vertices[1], point);
    return barycentric / (barycentric.x + barycentric.y + barycentric.z);
}

int main() {
    auto lightmapSize = uvec2(256);
    Scene scene(lightmapSize);
    scene.faces = loadMesh("resources/cornell_box.obj");

    auto texelSize = 1.0f / vec2(scene.lightmapPatches.size());
    for (auto& face : scene.faces) {
        vec2 min = glm::min(glm::min(face.lightmapUVs[0], face.lightmapUVs[1]), face.lightmapUVs[2]);
        vec2 max = glm::max(glm::max(face.lightmapUVs[0], face.lightmapUVs[1]), face.lightmapUVs[2]);

        uvec2 minTexel = uvec2(min / texelSize);
        uvec2 maxTexel = uvec2(max / texelSize);

        for (uint32_t y = minTexel.y; y <= maxTexel.y; y++) {
            for (uint32_t x = minTexel.x; x <= maxTexel.x; x++) {
                vec2 texelCenter = (vec2(x, y) + 0.5f) * texelSize;
                vec3 barycentric = barycentricCoordinates(face.lightmapUVs, texelCenter);
                if (glm::any(glm::lessThan(barycentric, vec3(0))))
                    continue;

                auto& patch = scene.lightmapPatches[uvec2(x, y)];
                patch.position = barycentric.x * face.vertices[0] + barycentric.y * face.vertices[1] + barycentric.z * face.vertices[2];
                patch.size = vec2(0);     // TODO
                patch.residue = vec3(0);  // TODO face.material->emission * patch.size;
                patch.face = &face;
            }
        }
    }

    for (uint32_t y = 0; y < lightmapSize.y; y++) {
        for (uint32_t x = 0; x < lightmapSize.x; x++) {
            auto& patch = scene.lightmapPatches[uvec2(x, y)];
            if (patch.face == nullptr)
                continue;

            scene.lightmapAccumulated[uvec2(x, y)] = patch.face->material->albedo;
        }
    }

    scene.lightmapAccumulated.save("lightmap.hdr");

    return EXIT_SUCCESS;
}
