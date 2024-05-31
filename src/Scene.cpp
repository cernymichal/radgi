#include "Scene.h"

void Scene::initialize(const uvec2& lightmapSize) {
    m_lightmapSize = lightmapSize;

    m_bvh.build();
    createPatches();
}

static f32 triangleArea(const vec2& a, const vec2& b, const vec2& c) {
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

static std::tuple<std::array<vec2, 4>, u8> faceTexelIntersection(const Face& face, const std::array<vec2, 4>& texelVertices) {
    std::array<vec2, 4> intersectionVertices;
    u8 intersectionVertexCount = 0;

    auto eps = 0.0001f;

    // add texel vertices if they are inside the face
    for (u32 i = 0; i < 4; i++) {
        vec3 barycentric = barycentricCoordinates(face.lightmapUVs, texelVertices[i]);
        if (glm::all(barycentric >= vec3(0 - eps)))
            intersectionVertices[intersectionVertexCount++] = texelVertices[i];
    }

    if (intersectionVertexCount == 4)  // all texel vertices are inside the face
        return {intersectionVertices, 4};

    // add face vertices inside the texel
    for (u32 i = 0; i < 3; i++) {
        if (glm::all(face.lightmapUVs[i] > texelVertices[0] - vec2(-eps)) && glm::all(face.lightmapUVs[i] < texelVertices[2] + vec2(-eps))) {
            // polygons with more than 4 vertices would be too complex to handle
            if (intersectionVertexCount == 4)
                return {intersectionVertices, 4};

            intersectionVertices[intersectionVertexCount++] = face.lightmapUVs[i];
        }
    }

    // add edge intersections
    for (u32 i = 0; i < 4; i++) {
        auto texelVertex = texelVertices[i];
        auto texelEdge = texelVertices[(i + 1) % 4] - texelVertex;

        for (u32 j = 0; j < 3; j++) {
            auto faceVertex = face.lightmapUVs[j];
            auto faceEdge = face.lightmapUVs[(j + 1) % 3] - faceVertex;

            vec2 params = lineIntersection(texelVertex, texelEdge, faceVertex, faceEdge);
            if (params.x > 0 - eps && params.x < 1 + eps && params.y > 0 - eps && params.y < 1 + eps) {
                // polygons with more than 4 vertices would be too complex to handle
                if (intersectionVertexCount == 4)
                    return {intersectionVertices, 4};

                auto intersection = texelVertex + params.x * texelEdge;
                intersectionVertices[intersectionVertexCount++] = intersection;
            }
        }
    }

    if (intersectionVertexCount < 3)
        return {intersectionVertices, 0};

    // sort the vertices to CCW order
    auto center = vec2(0);
    for (u32 i = 0; i < intersectionVertexCount; i++)
        center += intersectionVertices[i];
    center /= static_cast<f32>(intersectionVertexCount);
    for (u32 i = 0; i < intersectionVertexCount; i++) {  // bubble for 4 vertices here is ~2.5x faster than std::sort
        bool swapped = false;
        for (u32 j = 0; j < intersectionVertexCount - i - 1; j++) {
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

    // rasterize the scene faces into the lightmap
    auto texelSize = 1.0f / vec2(m_patches.size());
    for (auto& face : m_faces) {
        vec2 min = glm::min(glm::min(face.lightmapUVs[0], face.lightmapUVs[1]), face.lightmapUVs[2]);
        vec2 max = glm::max(glm::max(face.lightmapUVs[0], face.lightmapUVs[1]), face.lightmapUVs[2]);

        uvec2 minTexel = glm::clamp(uvec2(min / texelSize), uvec2(0), m_lightmapSize - uvec2(1));
        uvec2 maxTexel = glm::clamp(uvec2(max / texelSize), uvec2(0), m_lightmapSize - uvec2(1));

        for (u32 y = minTexel.y; y <= maxTexel.y; y++) {
            for (u32 x = minTexel.x; x <= maxTexel.x; x++) {
                std::array<vec2, 4> texelVertices = {
                    vec2(x, y) * texelSize,
                    vec2(x + 1, y) * texelSize,
                    vec2(x + 1, y + 1) * texelSize,
                    vec2(x, y + 1) * texelSize};

                auto [patchVertices, patchVertexCount] = faceTexelIntersection(face, texelVertices);

                if (patchVertexCount < 3)
                    continue;

                if (m_patches[uvec2(x, y)].face != nullptr) {
                    // TODO merge the patches - save all vertices in patch map and then create the geometry by merging equal vertices and collapsing edges

                    // patchVertices = texelVertices;
                    // patchVertexCount = 4;
                    continue;
                }

                // translate the texel vertices to world space
                bool invalidPatch = false;
                std::array<vec3, 4> patchVerticesWS;
                for (u32 i = 0; i < patchVertexCount; i++) {
                    vec3 barycentric = barycentricCoordinates(face.lightmapUVs, patchVertices[i]);
                    patchVerticesWS[i] = barycentric.x * face.vertices[0] + barycentric.y * face.vertices[1] + barycentric.z * face.vertices[2];

                    if (glm::any(glm::isnan(patchVerticesWS[i])))
                        invalidPatch = true;
                }
                if (invalidPatch)
                    continue;

                m_patches[uvec2(x, y)] = Patch(patchVertexCount, patchVerticesWS, &face);
            }
        }
    }
}

std::vector<Face> Scene::createPatchGeometry() const {
    std::vector<Face> geometry;
    for (u32 y = 0; y < m_lightmapSize.y; y++) {
        for (u32 x = 0; x < m_lightmapSize.x; x++) {
            auto& patch = m_patches[uvec2(x, y)];
            if (patch.face == nullptr)
                continue;

            if (patch.vertexCount >= 3) {
                auto& face = geometry.emplace_back();
                for (u32 i = 0; i < 3; i++)
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

void Scene::dilateLightmap(Texture<vec3>& lightmap, u32 radius) {
    Texture<bool> dilatedThisStep(m_lightmapSize);
    for (u32 i = 0; i < radius; i++) {
        dilatedThisStep.clear(false);
        for (i32 y = 0; y < m_lightmapSize.y; y++) {
            for (i32 x = 0; x < m_lightmapSize.x; x++) {
                auto idx = uvec2(x, y);

                if (m_patches[idx].face != nullptr)
                    continue;

                for (i32 dy = -1; dy <= 1; dy++) {
                    for (i32 dx = -1; dx <= 1; dx++) {
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
