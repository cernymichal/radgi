#include "IGISolver.h"

static vec3 randomPointOnPatch(const Patch& patch) {
    if (patch.vertexCount == 3) {
        auto u = random<f32>();
        auto v = random<f32>(0, 1.0f - u);
        auto w = 1.0f - u - v;
        return u * patch.vertices[0] + v * patch.vertices[1] + w * patch.vertices[2];
    }

    auto edge0 = patch.vertices[1] - patch.vertices[0];
    auto edge1 = patch.vertices[3] - patch.vertices[0];
    return patch.vertices[0] + random<f32>() * edge0 + random<f32>() * edge1;  // TODO this is wrong
}

f32 calculateFormFactor(const Patch& patchA, const Patch& patchB, const Scene& scene) {
    f32 F = 0;

    constexpr auto rayCount = 4;  // TODO make this a parameter
    for (u32 i = 0; i < rayCount; i++) {
        auto rayOrigin = randomPointOnPatch(patchA);
        auto rayTarget = randomPointOnPatch(patchB);

        // visibility test
        auto targetDistance = glm::length(rayTarget - rayOrigin);
        auto rayDirection = (rayTarget - rayOrigin) / targetDistance;

        Interval<f32> tInterval = {0.01f, targetDistance - 0.01f};  // leeway for shared edges passing through the lightmap

#define USE_BVH
#ifdef USE_BVH
        auto intersectionPredicate = [&](f32, const Face& face) {
            return face != *patchA.face && face != *patchB.face;
        };
        bool hit = scene.bvh().intersects(rayOrigin, rayDirection, tInterval, intersectionPredicate);
#else
        bool hit = false;
        for (const auto& face : scene.faces()) {
            if (glm::dot(-rayDirection, face.normal) <= 0)
                continue;

            if (face == *patchA.face || face == *patchB.face)
                continue;

            f32 t = rayTriangleIntersection(rayOrigin, rayDirection, face.vertices);

            if (!isnan(t) && tInterval.contains(t)) {
                hit = true;
                break;
            }
        }
#endif
        if (hit)  // visibility test failed
            continue;

        f32 r2 = glm::length2(rayTarget - rayOrigin);
        f32 cosines = glm::dot(rayDirection, patchA.face->normal) * glm::dot(-rayDirection, patchB.face->normal);
        f32 deltaF = cosines / (pi_v<f32> * r2);
        F += glm::max(deltaF, 0.0f);
    }

    return F / rayCount;
}
