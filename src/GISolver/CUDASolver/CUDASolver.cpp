#include "CUDASolver.h"

#include <cuda.h>

#include "CUDAStructs.h"

extern "C" vec3* solveRadiosityCUDA(u32 bounces, uvec2 lightmapSize, const CUDAStructs::Scene& sceneHost);

void CUDASolver::initialize(const Ref<const Scene>& scene) {
    IGISolver::initialize(scene);

    // TODO page locked memory
    // cudaHostAlloc
    // https://stackoverflow.com/questions/14807192/can-i-use-an-stdvector-as-a-facade-for-a-pre-allocated-raw-array

    m_lightmapSize = {static_cast<int>(scene->lightmapSize().x), static_cast<int>(scene->lightmapSize().y)};

    std::unordered_map<Ref<Material>, u16> materialIndices;
    m_materials.reserve(m_scene->materials().size());
    for (const auto& material : m_scene->materials()) {
        materialIndices[material] = static_cast<u16>(m_materials.size());

        auto& cudaMaterial = m_materials.emplace_back();
        cudaMaterial = {
            .albedo = material->albedo,
            .emission = material->emission};
    }

    m_faces.reserve(m_scene->faces().size());
    for (const auto& face : m_scene->faces()) {
        auto& cudaFace = m_faces.emplace_back();
        cudaFace = {
            .vertices = face.vertices,
            .normal = face.normal,
            .materialId = face.material != nullptr ? materialIndices[face.material] : static_cast<u16>(CUDAStructs::NULL_ID)};
    }

    m_patches.reserve(m_lightmapSize.x * m_lightmapSize.y);
    uvec2 patchIdx;
    for (patchIdx.y = 0; patchIdx.y < m_lightmapSize.y; patchIdx.y++) {
        for (patchIdx.x = 0; patchIdx.x < m_lightmapSize.x; patchIdx.x++) {
            const auto& patch = m_scene->patches()[patchIdx];

            auto& cudaPatch = m_patches.emplace_back();
            cudaPatch = {
                .vertices = patch.vertices,
                .vertexCount = patch.vertexCount,
                .area = patch.area,
                .faceId = patch.face != nullptr ? static_cast<u32>(patch.face - m_scene->faces().data()) : CUDAStructs::NULL_ID};
        }
    }

    m_bvhNodes.reserve(m_scene->bvh().m_nodes.size());
    for (const auto& node : m_scene->bvh().m_nodes) {
        auto& cudaNode = m_bvhNodes.emplace_back();

        cudaNode = {
            .aabb = node.aabb,
            .face = node.face};
    }
}

Texture<vec3> CUDASolver::solve() {
    CUDAStructs::Scene sceneHost = {
        .patches = m_patches.data(),
        .faces = m_faces.data(),
        .faceCount = static_cast<u32>(m_faces.size()),
        .materials = m_materials.data(),
        .materialCount = static_cast<u32>(m_materials.size()),
        .bvh = {
            .nodes = m_bvhNodes.data(),
            .nodeCount = static_cast<u32>(m_bvhNodes.size()),
        },
    };

    vec3* lightmap = solveRadiosityCUDA(m_bounces, m_lightmapSize, sceneHost);
    return Texture<vec3>(m_scene->lightmapSize(), std::move(lightmap));
}
