#include "Scene.h"

void Scene::initialize(const uvec2& lightmapSize) {
    m_lightmapSize = lightmapSize;

    m_bvh.build();
    createPatches();
}

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

std::tuple<std::array<vec2, 4>, uint8_t> faceTexelIntersection(const Face& face, const std::array<vec2, 4>& texelVertices) {
    std::array<vec2, 4> intersectionVertices;
    uint8_t intersectionVertexCount = 0;

    // add texel vertices if they are inside the face
    for (uint32_t i = 0; i < 4; i++) {
        vec3 barycentric = barycentricCoordinates(face.lightmapUVs, texelVertices[i]);
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

void Scene::createPatches() {
    m_patches = Texture<Patch>(m_lightmapSize);

    auto maxResiduePatch = uvec2(0, 0);
    float maxResidue2 = 0;  // squared magnitude

    // rasterize the scene faces into the lightmap
    auto texelSize = 1.0f / vec2(m_patches.size());
    for (auto& face : m_faces) {
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

                if (m_patches[uvec2(x, y)].face != nullptr) {
                    // TODO merge the patches somehow

                    patchVertices = texelVertices;
                    patchVertexCount = 4;
                }

                // translate the texel vertices to world space
                std::array<vec3, 4> patchVerticesWS;
                for (uint32_t i = 0; i < patchVertexCount; i++) {
                    vec3 barycentric = barycentricCoordinates(face.lightmapUVs, patchVertices[i]);
                    patchVerticesWS[i] = barycentric.x * face.vertices[0] + barycentric.y * face.vertices[1] + barycentric.z * face.vertices[2];
                }

                m_patches[uvec2(x, y)] = Patch(patchVertexCount, patchVerticesWS, &face);
            }
        }
    }
}

std::vector<Face> Scene::createPatchGeometry() const {
    std::vector<Face> geometry;
    for (uint32_t y = 0; y < m_lightmapSize.y; y++) {
        for (uint32_t x = 0; x < m_lightmapSize.x; x++) {
            auto& patch = m_patches[uvec2(x, y)];
            if (patch.face == nullptr)
                continue;

            if (patch.vertexCount >= 3) {
                auto& face = geometry.emplace_back();
                for (uint32_t i = 0; i < 3; i++)
                    face.vertices[i] = patch.vertices[i];
            }

            if (patch.vertexCount == 4) {
                auto& face = geometry.emplace_back();
                face.vertices[0] = patch.vertices[0];
                face.vertices[1] = patch.vertices[2];
                face.vertices[2] = patch.vertices[3];
            }
        }
    }
    return geometry;
}

void Scene::dilateLightmap(Texture<vec3>& lightmap, uint32_t radius) {
    Texture<bool> dilatedThisStep(m_lightmapSize);
    for (uint32_t i = 0; i < radius; i++) {
        dilatedThisStep.clear(false);
        for (uint32_t y = 0; y < m_lightmapSize.y; y++) {
            for (uint32_t x = 0; x < m_lightmapSize.x; x++) {
                auto idx = uvec2(x, y);

                if (m_patches[idx].face != nullptr)
                    continue;

                for (int32_t dy = -1; dy <= 1; dy++) {
                    for (int32_t dx = -1; dx <= 1; dx++) {
                        if (abs(dx) == 1 && abs(dy) == 1)
                            continue;

                        if (x + dx < 0 || x + dx >= m_lightmapSize.x || y + dy < 0 || y + dy >= m_lightmapSize.y)
                            continue;

                        auto sampleIdx = uvec2(x + dx, y + dy);

                        if (m_patches[sampleIdx].face != nullptr && !dilatedThisStep[sampleIdx]) {
                            lightmap[idx] = lightmap[sampleIdx];
                            m_patches[idx].face = m_patches[sampleIdx].face;
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
