#include "RadiositySolver.h"

static float triangleArea(const vec2& a, const vec2& b, const vec2& c) {
    vec2 edge0 = b - a;
    vec2 edge1 = c - a;
    return glm::cross(edge0, edge1) / 2;
}

static vec3 barycentricCoordinates(const std::array<vec2, 3>& vertices, const vec2& point) {
    auto area = triangleArea(vertices[0], vertices[1], vertices[2]);
    vec3 barycentric;
    barycentric.x = triangleArea(vertices[1], vertices[2], point);
    barycentric.y = triangleArea(vertices[2], vertices[0], point);
    barycentric.z = triangleArea(vertices[0], vertices[1], point);
    return barycentric / area;
}

static vec3 randomPointOnPatch(const Patch& patch) {
    if (patch.vertexCount == 3) {
        auto u = random<float>();
        auto v = random<float>(0, 1.0f - u);
        auto w = 1.0f - u - v;
        return u * patch.vertices[0] + v * patch.vertices[1] + w * patch.vertices[2];
    }

    auto edge0 = patch.vertices[1] - patch.vertices[0];
    auto edge1 = patch.vertices[3] - patch.vertices[0];
    return patch.vertices[0] + random<float>() * edge0 + random<float>() * edge1;
}

float RadiositySolver::calculateFormFactor(const Patch& patchA, const Patch& patchB) {
    float F = 0;

    constexpr auto rayCount = 8;  // TODO make this a parameter
    for (uint32_t i = 0; i < rayCount; i++) {
        auto rayOrigin = randomPointOnPatch(patchA);
        auto rayTarget = randomPointOnPatch(patchB);

        // visibility test
        auto targetDistance = glm::length(rayTarget - rayOrigin);
        auto rayDirection = (rayTarget - rayOrigin) / targetDistance;

        if (glm::dot(rayDirection, patchA.face->normal) <= 0 || glm::dot(-rayDirection, patchB.face->normal) <= 0)
            continue;

        bool hit = false;
        for (const auto& face : m_scene->faces) {
            if (glm::dot(-rayDirection, face.normal) <= 0)
                continue;

            if (face == *patchA.face || face == *patchB.face)
                continue;

            auto t = rayTriangleIntersection(rayOrigin, rayDirection, narrowToTriangle(face.vertices));
            if (std::isnan(t))
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

std::tuple<std::array<vec2, 4>, uint8_t> faceTexelIntersection(const Face& face, const std::array<vec2, 4>& texelVertices) {
    std::array<vec2, 4> intersectionVertices;
    uint8_t intersectionVertexCount = 0;

    // add texel vertices if they are inside the face
    for (uint32_t i = 0; i < 4; i++) {
        vec3 barycentric = barycentricCoordinates(narrowToTriangle(face.lightmapUVs), texelVertices[i]);
        if (glm::all(barycentric >= vec3(0)))
            intersectionVertices[intersectionVertexCount++] = texelVertices[i];
    }

    if (intersectionVertexCount == 4)  // all texel vertices are inside the face
        return {texelVertices, 4};

    // add edge intersections
    for (uint32_t i = 0; i < 4; i++) {
        auto texelVertex = texelVertices[i];
        auto texelEdge = texelVertices[(i + 1) % 4] - texelVertex;

        for (uint32_t j = 0; j < 3; j++) {
            auto faceVertex = face.lightmapUVs[j];
            auto faceEdge = face.lightmapUVs[(j + 1) % 3] - faceVertex;

            vec2 params = lineIntersection(texelVertex, texelEdge, faceVertex, faceEdge);
            if (params.x > 0 && params.x < 1 && params.y >= 0 && params.y <= 1) {
                // polygons with more than 4 vertices would be too complex to handle
                if (intersectionVertexCount == 4)
                    return {texelVertices, 4};

                auto intersection = texelVertex + params.x * texelEdge;
                intersectionVertices[intersectionVertexCount++] = intersection;
            }
        }
    }

    // add face vertices inside the texel
    for (uint32_t i = 0; i < 3; i++) {
        if (glm::all(face.lightmapUVs[i] > texelVertices[0]) && glm::all(face.lightmapUVs[i] < texelVertices[2])) {
            // polygons with more than 4 vertices would be too complex to handle
            if (intersectionVertexCount == 4)
                return {texelVertices, 4};

            intersectionVertices[intersectionVertexCount++] = face.lightmapUVs[i];
        }
    }

    if (intersectionVertexCount < 3)
        return {{}, 0};

    // sort the vertices to CCW order
    auto center = vec2(0);
    for (uint32_t i = 0; i < intersectionVertexCount; i++)
        center += intersectionVertices[i];
    center /= static_cast<float>(intersectionVertexCount);
    for (uint32_t i = 0; i < intersectionVertexCount; i++) {  // bubble for 4 vertices here is ~2.5x faster than std::sort
        bool swapped = false;
        for (uint32_t j = 0; j < intersectionVertexCount - i - 1; j++) {
            auto& a = intersectionVertices[j];
            auto& b = intersectionVertices[j + 1];
            if (atan2(a.x - center.x, a.y - center.y) > atan2(b.x - center.x, b.y - center.y)) {
                std::swap(a, b);
                swapped = true;
            }
        }
        if (!swapped)
            break;
    }

    return {intersectionVertices, intersectionVertexCount};
}

void RadiositySolver::initialize(const Ref<Scene>& scene) {
    m_scene = scene;

    auto maxResiduePatch = uvec2(0, 0);
    float maxResidue2 = 0;  // squared magnitude

    // rasterize the scene faces into the lightmap
    auto texelSize = 1.0f / vec2(m_lightmapPatches.size());
    for (auto& face : m_scene->faces) {
        assert(face.vertexCount == 3);

        vec2 min = glm::min(glm::min(face.lightmapUVs[0], face.lightmapUVs[1]), face.lightmapUVs[2]);
        vec2 max = glm::max(glm::max(face.lightmapUVs[0], face.lightmapUVs[1]), face.lightmapUVs[2]);

        uvec2 minTexel = uvec2(min / texelSize);
        uvec2 maxTexel = uvec2(max / texelSize);

        for (uint32_t y = minTexel.y; y <= maxTexel.y; y++) {
            for (uint32_t x = minTexel.x; x <= maxTexel.x; x++) {
                std::array<vec2, 4> texelVertices = {
                    vec2(x, y) * texelSize,
                    vec2(x + 1, y) * texelSize,
                    vec2(x + 1, y + 1) * texelSize,
                    vec2(x, y + 1) * texelSize};

                auto [patchVertices, patchVertexCount] = faceTexelIntersection(face, texelVertices);

                if (patchVertexCount < 3)
                    continue;

                if (m_lightmapPatches[uvec2(x, y)].face != nullptr) {
                    // TODO merge the patches somehow

                    patchVertices = texelVertices;
                    patchVertexCount = 4;
                }

                // translate the texel vertices to world space
                std::array<vec3, 4> patchVerticesWS;
                for (uint32_t i = 0; i < patchVertexCount; i++) {
                    vec3 barycentric = barycentricCoordinates(narrowToTriangle(face.lightmapUVs), patchVertices[i]);
                    patchVerticesWS[i] = barycentric.x * face.vertices[0] + barycentric.y * face.vertices[1] + barycentric.z * face.vertices[2];
                }

                m_lightmapPatches[uvec2(x, y)] = Patch(patchVertexCount, patchVerticesWS, &face);

                auto residue = m_lightmapPatches[uvec2(x, y)].residue;
                m_lightmapAccumulated[uvec2(x, y)] = residue;

                // TODO this isnt updated properly after because of overwriting in the lightmap
                auto residueMagnitude2 = glm::length2(residue);
                if (residueMagnitude2 > maxResidue2) {
                    maxResidue2 = residueMagnitude2;
                    maxResiduePatch = uvec2(x, y);
                }
            }
        }
    }

    m_maxResiduePatchIdx = maxResiduePatch;
}

void RadiositySolver::solveShooting(float residueThreshold) {
    for (uint32_t i = 0;; i++) {
        auto residue = shoot(m_maxResiduePatchIdx, residueThreshold);
        if (residue <= residueThreshold)
            break;

        if (i % 100 == 0)
            LOG(std::format("{}: residue={:0.4f}", i, residue));
    }
}

void RadiositySolver::solveGathering(uint32_t iterations) {
    for (uint32_t i = 0; i < iterations; i++) {
        auto sourceIdx = uvec2(0, 0);
        for (sourceIdx.y = 0; sourceIdx.y < m_lightmapSize.y; sourceIdx.y++) {
            for (sourceIdx.x = 0; sourceIdx.x < m_lightmapSize.x; sourceIdx.x++) {
                gather(sourceIdx);
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

            // check if the patches are facing each other
            auto sightLine = glm::normalize(receiver.center - source.center);
            if (glm::dot(sightLine, source.face->normal) <= 0 || glm::dot(-sightLine, receiver.face->normal) <= 0)
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

float RadiositySolver::gather(uvec2 destinationIdx) {
    // TODO implement
    return 0;
}

void RadiositySolver::dilateLightmap(uint32_t radius) {
    Texture<bool> dilatedThisStep(m_lightmapSize);
    for (uint32_t i = 0; i < radius; i++) {
        dilatedThisStep.clear(false);
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

                        if (m_lightmapPatches[sampleIdx].face != nullptr && !dilatedThisStep[sampleIdx]) {
                            m_lightmapAccumulated[idx] = m_lightmapAccumulated[sampleIdx];
                            m_lightmapPatches[idx].face = m_lightmapPatches[sampleIdx].face;
                            dilatedThisStep[idx] = true;
                            break;
                        }
                    }
                    if (dilatedThisStep[idx])
                        break;
                }
            }
        }
    }
}

std::vector<Face> RadiositySolver::createPatchGeometry() const {
    std::vector<Face> geometry;
    for (uint32_t y = 0; y < m_lightmapSize.y; y++) {
        for (uint32_t x = 0; x < m_lightmapSize.x; x++) {
            auto& patch = m_lightmapPatches[uvec2(x, y)];
            if (patch.face == nullptr)
                continue;

            auto& face = geometry.emplace_back();
            face.vertexCount = patch.vertexCount;
            for (uint32_t i = 0; i < patch.vertexCount; i++)
                face.vertices[i] = patch.vertices[i];
        }
    }
    return geometry;
}
