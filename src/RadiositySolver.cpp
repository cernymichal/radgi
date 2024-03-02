#include "RadiositySolver.h"

static float triangleArea(const vec2& a, const vec2& b, const vec2& c) {
    vec2 edge0 = b - a;
    vec2 edge1 = c - a;
    return (edge0.x * edge1.y - edge1.x * edge0.y) / 2;
}

static vec3 barycentricCoordinates(const vec2 vertices[3], const vec2& point) {
    auto area = triangleArea(vertices[0], vertices[1], vertices[2]);
    vec3 barycentric;
    barycentric.x = triangleArea(vertices[1], vertices[2], point);
    barycentric.y = triangleArea(vertices[2], vertices[0], point);
    barycentric.z = triangleArea(vertices[0], vertices[1], point);
    return barycentric / area;
}

static std::pair<bool, float> rayTriangleIntersection(const vec3& rayOrigin, const vec3& rayDirection, const vec3 vertices[3], const vec3& normal) {
    constexpr auto epsilon = 0.0001f;

    // Möller–Trumbore intersection algorithm

    auto edge1 = vertices[1] - vertices[0];
    auto edge2 = vertices[2] - vertices[0];
    auto P = glm::cross(rayDirection, edge2);
    auto determinant = glm::dot(edge1, P);

    // if the determinant is negative, the triangle is back facing
    // if the determinant is close to 0, the ray misses the triangle
    if (determinant < epsilon)
        return {false, 0};

    auto determinantInv = 1.0f / determinant;
    auto T = rayOrigin - vertices[0];
    auto u = glm::dot(T, P) * determinantInv;
    if (u < 0 || u > 1)
        return {false, 0};

    auto Q = glm::cross(T, edge1);
    auto v = glm::dot(rayDirection, Q) * determinantInv;
    if (v < 0 || u + v > 1)
        return {false, 0};

    auto t = glm::dot(edge2, Q) * determinantInv;
    if (t < 0)
        return {false, 0};

    return {true, t};
}

float RadiositySolver::calculateVisibility(const Patch& source, const Patch& destination) {
    auto destinationDistance = glm::length(destination.center - source.center);

    auto rayOrigin = source.center;
    auto rayDirection = (destination.center - source.center) / destinationDistance;
    for (const auto& face : m_scene->faces) {
        if (face == *source.face || face == *destination.face)
            continue;

        auto [intersects, t] = rayTriangleIntersection(rayOrigin, rayDirection, face.vertices, face.normal);
        if (!intersects)
            continue;

        if (t < destinationDistance - 0.001f) // leeway for shared edges passing through the lightmap
            return 0;
    }

    return 1;
}

float RadiositySolver::calculateFormFactor(const Patch& source, const Patch& destination) {
    auto direction = destination.center - source.center;
    auto distance = glm::length(direction);
    direction /= distance;

    auto cosine = glm::dot(source.face->normal, direction) * glm::dot(destination.face->normal, -direction);
    auto V = calculateVisibility(source, destination);
    float F = cosine / (PI * distance * distance) * V;
    return glm::max(0.0f, F);
}

void RadiositySolver::initialize(const Ref<Scene>& scene) {
    m_scene = scene;

    auto texelSize = 1.0f / vec2(m_lightmapPatches.size());
    for (auto& face : m_scene->faces) {
        vec2 min = glm::min(glm::min(face.lightmapUVs[0], face.lightmapUVs[1]), face.lightmapUVs[2]);
        vec2 max = glm::max(glm::max(face.lightmapUVs[0], face.lightmapUVs[1]), face.lightmapUVs[2]);

        uvec2 minTexel = uvec2(min / texelSize);
        uvec2 maxTexel = uvec2(max / texelSize);

        for (uint32_t y = minTexel.y; y <= maxTexel.y; y++) {
            for (uint32_t x = minTexel.x; x <= maxTexel.x; x++) {
                vec2 texelVertices[4] = {
                    vec2(x, y) * texelSize,
                    vec2(x + 1, y) * texelSize,
                    vec2(x + 1, y + 1) * texelSize,
                    vec2(x, y + 1) * texelSize};

                vec3 patchVertices[4];
                uint32_t verticesInsideFace = 0;
                for (uint32_t i = 0; i < 4; i++) {
                    vec3 barycentric = barycentricCoordinates(face.lightmapUVs, texelVertices[i]);
                    patchVertices[i] = barycentric.x * face.vertices[0] + barycentric.y * face.vertices[1] + barycentric.z * face.vertices[2];

                    if (glm::all(glm::greaterThanEqual(barycentric, vec3(0))))
                        verticesInsideFace++;
                }

                if (verticesInsideFace == 0)  // texel is outside the face
                    continue;

                m_lightmapPatches[uvec2(x, y)] = Patch(patchVertices, &face);
                m_lightmapAccumulated[uvec2(x, y)] = m_lightmapPatches[uvec2(x, y)].residue;
            }
        }
    }
}

void RadiositySolver::solve(float residueEpsilon, uint32_t iterations) {
    for (uint32_t i = 0; i < iterations; i++) {
        auto residue = shoot(residueEpsilon);
        if (residue <= residueEpsilon)
            break;
    }
}

float RadiositySolver::shoot(float residueEpsilon) {
    // find the patch with the largest residue
    auto& shootingPatch = m_lightmapPatches[uvec2(0, 0)];
    auto maxResidue = glm::length2(shootingPatch.residue);
    for (uint32_t y = 0; y < m_lightmapSize.y; y++) {
        for (uint32_t x = 0; x < m_lightmapSize.x; x++) {
            auto& patch = m_lightmapPatches[uvec2(x, y)];
            if (patch.face == nullptr)
                continue;

            auto residueMagnitude = glm::length2(patch.residue);
            if (residueMagnitude >= maxResidue) {
                maxResidue = residueMagnitude;
                shootingPatch = patch;
            }
        }
    }
    maxResidue = glm::sqrt(maxResidue);

    if (maxResidue <= residueEpsilon)
        return 0;  // nothing to solve

    // shoot to other patches
    for (uint32_t y = 0; y < m_lightmapSize.y; y++) {
        for (uint32_t x = 0; x < m_lightmapSize.x; x++) {
            auto& destinationPatch = m_lightmapPatches[uvec2(x, y)];
            if (destinationPatch.face == nullptr || destinationPatch == shootingPatch)
                continue;

            auto F = calculateFormFactor(shootingPatch, destinationPatch);
            // auto dA = destinationPatch.area / shootingPatch.area;
            auto radDelta = F * shootingPatch.residue * destinationPatch.face->material->albedo;
            destinationPatch.residue += radDelta;
            m_lightmapAccumulated[uvec2(x, y)] += radDelta;
        }
    }

    shootingPatch.residue = vec3(0);
    LOG("residue=" << maxResidue);
    return maxResidue;  // return the amount of light shot
}
