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

static std::pair<bool, float> rayTriangleIntersection(const vec3& rayOrigin, const vec3& rayDirection, const vec3 vertices[3]) {
    constexpr auto epsilon = 0.001f;

    // M�ller�Trumbore intersection algorithm

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

static vec3 randomPointOnPatch(const Patch& patch) {
    auto edge0 = patch.vertices[1] - patch.vertices[0];
    auto edge1 = patch.vertices[3] - patch.vertices[0];
    return patch.vertices[0] + random<float>() * edge0 + random<float>() * edge1;
}

float RadiositySolver::calculateFormFactor(const Patch& patchA, const Patch& patchB) {
    float F = 0;

    constexpr auto rayCount = 8;
    for (uint32_t i = 0; i < rayCount; i++) {
        auto rayOrigin = randomPointOnPatch(patchA);
        auto rayTarget = randomPointOnPatch(patchB);

        // visibility test
        auto targetDistance = glm::length(rayTarget - rayOrigin);
        auto rayDirection = (rayTarget - rayOrigin) / targetDistance;
        bool hit = false;
        for (const auto& face : m_scene->faces) {
            if (glm::dot(rayDirection, face.normal) > 0.01f)
                continue;

            if (face == *patchA.face || face == *patchB.face)
                continue;

            auto [intersects, t] = rayTriangleIntersection(rayOrigin, rayDirection, face.vertices);
            if (!intersects)
                continue;

            if (t < targetDistance - 0.01f) {  // leeway for shared edges passing through the lightmap
                hit = true;
                break;
            }
        }

        if (hit)  // visibility test failed
            continue;

        auto r2 = glm::length2(rayTarget - rayOrigin);
        auto cosines = glm::dot(rayDirection, patchA.face->normal) * glm::dot(-rayDirection, patchB.face->normal);
        auto deltaF = cosines * patchB.area / (PI * r2);  // + patchB.area / rayCount);

        if (deltaF > 0)
            F += deltaF;
    }

    return F / rayCount;
}

void RadiositySolver::initialize(const Ref<Scene>& scene) {
    m_scene = scene;

    auto maxResiduePatch = uvec2(0, 0);
    float maxResidue2 = 0;  // squared magnitude

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
                auto residue = m_lightmapPatches[uvec2(x, y)].residue;
                m_lightmapAccumulated[uvec2(x, y)] = residue;

                auto residueMagnitude2 = glm::length2(residue);
                if (residueMagnitude2 > maxResidue2) {
                    maxResidue2 = residueMagnitude2;
                    maxResiduePatch = uvec2(x, y);
                }
            }
        }

        m_maxResiduePatchIdx = maxResiduePatch;
    }
}

void RadiositySolver::solveProgressive(float residueThreshold) {
    for (uint32_t i = 0;; i++) {
        auto residue = shoot(m_maxResiduePatchIdx, residueThreshold);
        if (residue <= residueThreshold)
            break;

        if (i % 1000 == 0)
            LOG(std::format("{}: residue={:0.4f}", i, residue));
    }
}

void RadiositySolver::solveUniform(uint32_t iterations) {
    for (uint32_t i = 0; i < iterations; i++) {
        auto sourceIdx = uvec2(0, 0);
        for (sourceIdx.y = 0; sourceIdx.y < m_lightmapSize.y; sourceIdx.y++) {
            for (sourceIdx.x = 0; sourceIdx.x < m_lightmapSize.x; sourceIdx.x++) {
                shoot(sourceIdx, 0);
            }
        }
        LOG(std::format("Iteration {}/{}", i + 1, iterations));
    }
}

float RadiositySolver::shoot(uvec2 sourceIdx, float residueThreshold) {
    auto& source = m_lightmapPatches[sourceIdx];
    auto shotRad = glm::length(source.residue);

    if (source.face == nullptr || shotRad <= residueThreshold)
        return 0;  // nothing to solve

    auto maxResiduePatch = uvec2(0, 0);
    float maxResidue2 = 0;  // squared magnitude

    float reflectedRad = 0;

    // shoot to other patches
    auto receiverIdx = uvec2(0, 0);
    for (receiverIdx.y = 0; receiverIdx.y < m_lightmapSize.y; receiverIdx.y++) {
        for (receiverIdx.x = 0; receiverIdx.x < m_lightmapSize.x; receiverIdx.x++) {
            auto& receiver = m_lightmapPatches[receiverIdx];
            if (receiver.face == nullptr || receiver == source)
                continue;

            // check for max residue canditate before possible skipping
            auto residueMagnitude2 = glm::length2(receiver.residue);
            if (residueMagnitude2 > maxResidue2) {
                maxResidue2 = residueMagnitude2;
                maxResiduePatch = receiverIdx;
            }

            if (receiver.face == source.face)
                continue;

            auto F = calculateFormFactor(source, receiver);
            if (F == 0)
                continue;

            auto deltaRad = receiver.face->material->albedo * source.residue * F * source.area / receiver.area;
            receiver.residue += deltaRad;
            m_lightmapAccumulated[receiverIdx] += deltaRad;
            reflectedRad += glm::length(deltaRad);

            // check for max residue canditate
            residueMagnitude2 = glm::length2(receiver.residue);
            if (residueMagnitude2 > maxResidue2) {
                maxResidue2 = residueMagnitude2;
                maxResiduePatch = receiverIdx;
            }
        }
    }

    source.residue = vec3(0);
    m_maxResiduePatchIdx = maxResiduePatch;

    // LOG(std::format("shotRad={:.04f} reflectedRadPer={:.02f}", shotRad, reflectedRad / shotRad));
    return shotRad;  // return the amount of light shot
}

void RadiositySolver::addPadding(uint32_t radius) {
    Texture<bool> extrapolatedThisIter(m_lightmapSize);
    for (uint32_t i = 0; i < radius; i++) {
        extrapolatedThisIter.clear(false);
        for (uint32_t y = 0; y < m_lightmapSize.y; y++) {
            for (uint32_t x = 0; x < m_lightmapSize.x; x++) {
                auto idx = uvec2(x, y);

                if (m_lightmapPatches[idx].face != nullptr)
                    continue;

                for (int32_t dy = -1; dy <= 1; dy++) {
                    for (int32_t dx = -1; dx <= 1; dx++) {
                        if (abs(dx) == 1 && abs(dy) == 1)
                            continue;

                        if (x + dx < 0 || x + dx >= m_lightmapSize.x || y + dy < 0 || y + dy >= m_lightmapSize.y)
                            continue;

                        auto sampleIdx = uvec2(x + dx, y + dy);

                        if (m_lightmapPatches[sampleIdx].face != nullptr && !extrapolatedThisIter[sampleIdx]) {
                            m_lightmapAccumulated[idx] = m_lightmapAccumulated[sampleIdx];
                            m_lightmapPatches[idx].face = m_lightmapPatches[sampleIdx].face;
                            extrapolatedThisIter[idx] = true;
                            break;
                        }
                    }
                    if (extrapolatedThisIter[idx])
                        break;
                }
            }
        }
    }
}
