#include "CUDASolver.h"

#include "CUDAStructs.h"

extern "C" float3* solveRadiosityCUDA(uint32_t bounces, int2 lightmapSize, CUDAStructs::Patch* patches, size_t sceneFaceCount, CUDAStructs::Face* sceneFaces, size_t materialCount, CUDAStructs::Material* materials);

void CUDASolver::initialize(const Ref<const Scene>& scene) {
    IGISolver::initialize(scene);

    m_lightmapSize = {static_cast<int>(scene->lightmapSize().x), static_cast<int>(scene->lightmapSize().y)};

    std::unordered_map<Ref<Material>, uint16_t> materialIndices;
    m_materials.reserve(m_scene->materials().size());
    for (const auto& material : m_scene->materials()) {
        materialIndices[material] = static_cast<uint16_t>(m_materials.size());
        auto& cudaMaterial = m_materials.emplace_back();

        cudaMaterial.albedo = reinterpret_cast<const float3&>(material->albedo);
        cudaMaterial.emission = reinterpret_cast<const float3&>(material->emission);
    }

    std::unordered_map<Face*, uint32_t> faceIndices;
    m_faces.reserve(m_scene->faces().size());
    for (const auto& face : m_scene->faces()) {
        faceIndices[const_cast<Face*>(&face)] = static_cast<uint32_t>(m_faces.size());
        auto& cudaFace = m_faces.emplace_back();

        for (size_t i = 0; i < 3; i++)
            cudaFace.vertices[i] = reinterpret_cast<const float3&>(face.vertices[i]);
        cudaFace.normal = reinterpret_cast<const float3&>(face.normal);
        cudaFace.materialId = face.material != nullptr ? materialIndices[face.material] : CUDAStructs::NULL_ID;
    }

    m_patches.reserve(m_lightmapSize.x * m_lightmapSize.y);
    uvec2 patchIdx;
    for (patchIdx.y = 0; patchIdx.y < m_lightmapSize.y; patchIdx.y++) {
        for (patchIdx.x = 0; patchIdx.x < m_lightmapSize.x; patchIdx.x++) {
            const auto& patch = m_scene->patches()[patchIdx];
            auto& cudaPatch = m_patches.emplace_back();

            for (size_t i = 0; i < patch.vertexCount; i++)
                cudaPatch.vertices[i] = reinterpret_cast<const float3&>(patch.vertices[i]);
            cudaPatch.vertexCount = patch.vertexCount;
            cudaPatch.center = reinterpret_cast<const float3&>(patch.center);
            cudaPatch.area = patch.area;
            cudaPatch.faceId = patch.face != nullptr ? faceIndices[patch.face] : CUDAStructs::NULL_ID;
        }
    }
}

Texture<vec3> CUDASolver::solve() {
    float3* lightmap = solveRadiosityCUDA(m_bounces, m_lightmapSize, m_patches.data(), m_faces.size(), m_faces.data(), m_materials.size(), m_materials.data());
    return Texture<vec3>(m_scene->lightmapSize(), reinterpret_cast<vec3*>(lightmap));
}
