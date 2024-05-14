#include <stdio.h>

#include <chrono>

#include "CUDAStructs.h"
#include "utils.cuh"

using namespace CUDAStructs;

__device__ vec3 randomPointOnPatch(const Patch& patch, RNG& rng) {
    if (patch.vertexCount == 3) {
        auto u = rng();
        auto v = rng(0, 1.0f - u);
        auto w = 1.0f - u - v;
        return u * patch.vertices[0] + v * patch.vertices[1] + w * patch.vertices[2];
    }

    auto edge0 = patch.vertices[1] - patch.vertices[0];
    auto edge1 = patch.vertices[3] - patch.vertices[0];
    return patch.vertices[0] + rng() * edge0 + rng() * edge1;  // TODO this is wrong
}

__device__ f32 calculateFormFactor(const Patch& patchA, const Patch& patchB, const Scene& scene, RNG& rng) {
    f32 F = 0;

    constexpr auto rayCount = 4;  // TODO make this a parameter
    for (u32 i = 0; i < rayCount; i++) {
        auto rayOrigin = randomPointOnPatch(patchA, rng);
        auto rayTarget = randomPointOnPatch(patchB, rng);

        // visibility test
        auto targetDistance = glm::length(rayTarget - rayOrigin);
        auto rayDirection = (rayTarget - rayOrigin) / targetDistance;

        Interval<f32> tInterval = {0.01f, targetDistance - 0.01f};  // leeway for shared edges passing through the lightmap

#define USE_BVH
#ifdef USE_BVH
        u32 excludeFaces[] = {patchA.faceId, patchB.faceId};
        bool hit = intersectsBVH(scene, rayOrigin, rayDirection, tInterval, excludeFaces);
#else
        bool hit = false;
        for (u32 i = 0; i < scene.faceCount; i++) {
            auto& face = scene.faces[i];
            if (glm::dot(-rayDirection, face.normal) <= 0)
                continue;

            if (i == patchA.faceId || i == patchB.faceId)
                continue;

            auto t = rayTriangleIntersection(rayOrigin, rayDirection, face.vertices);

            if (!isnan(t) && tInterval.contains(t)) {
                hit = true;
                break;
            }
        }
#endif

        if (hit)  // visibility test failed
            continue;

        auto ray = rayTarget - rayOrigin;
        auto r2 = glm::dot(ray, ray);
        auto cosines = glm::dot(rayDirection, scene.faces[patchA.faceId].normal) * glm::dot(-rayDirection, scene.faces[patchB.faceId].normal);
        f32 deltaF = cosines / (PI * r2);
        F += glm::max(deltaF, 0.0f);
    }

    return F / rayCount;
}

__device__ __forceinline__ void atomicAdd(vec3* target, const vec3& value) {
    atomicAdd(&target->x, value.x);
    atomicAdd(&target->y, value.y);
    atomicAdd(&target->z, value.z);
}

__global__ void gather(const uvec2 lightmapSize, vec3* lightmap, const vec3* residues, vec3* nextResidues, const Scene scene, const u32 rngSeed, u32 threadQuota) {
    u64 threadId = blockIdx.x * blockDim.x + threadIdx.x;
    u64 startCombinationIdx = threadId * threadQuota;
    u64 patchCount = lightmapSize.x * lightmapSize.y;

    glm::vec<2, u64> patches;
    patches.y = floor((sqrt(8.0 * startCombinationIdx + 1) + 1) / 2);
    const auto prev_y = patches.y - 1;
    patches.x = startCombinationIdx - prev_y * (prev_y - 1) / 2 - prev_y;

    RNG rng(rngSeed + threadId);

    for (i32 i = 0; i < threadQuota; i++) {
        if (i != 0)
            patches.x++;
        if (patches.x >= patches.y) {
            patches.x = 0;
            patches.y++;
        }

        if (patches.x >= patchCount || patches.y >= patchCount)
            return;

        auto& patchA = scene.patches[patches.x];
        auto& patchB = scene.patches[patches.y];
        if (patchA.faceId == NULL_ID || patchB.faceId == NULL_ID)
            continue;

        vec3 patchAResidue = residues[patches.x];
        vec3 patchBResidue = residues[patches.y];
        if (patchAResidue == vec3(0) && patchBResidue == vec3(0))
            continue;

        // check if the patches are facing each other
        auto sightLine = glm::normalize(patchB.vertices[0] - patchA.vertices[0]);
        if (glm::dot(sightLine, scene.faces[patchA.faceId].normal) <= 0 || glm::dot(-sightLine, scene.faces[patchB.faceId].normal) <= 0)
            continue;

        auto F = calculateFormFactor(patchA, patchB, scene, rng);
        if (F == 0)
            continue;

        auto deltaRadA = scene.materials[scene.faces[patchA.faceId].materialId].albedo * patchBResidue * F * patchB.area;
        auto deltaRadB = scene.materials[scene.faces[patchB.faceId].materialId].albedo * patchAResidue * F * patchA.area;

        atomicAdd(&nextResidues[patches.x], deltaRadA);
        atomicAdd(&lightmap[patches.x], deltaRadA);
        atomicAdd(&nextResidues[patches.y], deltaRadB);
        atomicAdd(&lightmap[patches.y], deltaRadB);
    }
}

__global__ void initTextures(uvec2 lightmapSize, vec3* lightmap, vec3* residues, vec3* nextResidues, const Scene scene) {
    auto texelST = uvec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (texelST.x >= lightmapSize.x || texelST.y >= lightmapSize.y)
        return;

    auto patchIdx = texelST.y * lightmapSize.x + texelST.x;
    auto& patch = scene.patches[patchIdx];

    vec3 residue;
    if (patch.faceId != NULL_ID)
        residue = scene.materials[scene.faces[patch.faceId].materialId].emission;
    else
        residue = vec3(0);

    lightmap[patchIdx] = residue;
    residues[patchIdx] = residue;
    nextResidues[patchIdx] = vec3(0);
}

extern "C" vec3* solveRadiosityCUDA(u32 bounces, uvec2 lightmapSize, const Scene& sceneHost) {
    Scene sceneDevice = sceneHost;

    // upload scene data to the device
    cudaMalloc(&sceneDevice.faces, sceneHost.faceCount * sizeof(Face));
    cudaMemcpy(sceneDevice.faces, sceneHost.faces, sceneHost.faceCount * sizeof(Face), cudaMemcpyHostToDevice);

    cudaMalloc(&sceneDevice.patches, lightmapSize.x * lightmapSize.y * sizeof(Patch));
    cudaMemcpy(sceneDevice.patches, sceneHost.patches, lightmapSize.x * lightmapSize.y * sizeof(Patch), cudaMemcpyHostToDevice);

    cudaMalloc(&sceneDevice.materials, sceneHost.materialCount * sizeof(Material));
    cudaMemcpy(sceneDevice.materials, sceneHost.materials, sceneHost.materialCount * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&sceneDevice.bvh.nodes, sceneHost.bvh.nodeCount * sizeof(BVH::Node));
    cudaMemcpy(sceneDevice.bvh.nodes, sceneHost.bvh.nodes, sceneHost.bvh.nodeCount * sizeof(BVH::Node), cudaMemcpyHostToDevice);

    // allocate work buffers
    vec3* nextResiduesDevice;
    cudaMalloc(&nextResiduesDevice, lightmapSize.x * lightmapSize.y * sizeof(vec3));

    vec3* lightmapDevice;
    cudaMalloc(&lightmapDevice, lightmapSize.x * lightmapSize.y * sizeof(vec3));

    vec3* residuesDevice;
    cudaMalloc(&residuesDevice, lightmapSize.x * lightmapSize.y * sizeof(vec3));

    checkCUDAError(cudaPeekAtLastError());
    checkCUDAError(cudaDeviceSynchronize());

    dim3 blockSize(16, 16);  // TODO optimize for SM occupancy
    dim3 blocks(ceil(lightmapSize.x / static_cast<f64>(blockSize.x)), ceil(lightmapSize.y / static_cast<f64>(blockSize.y)));

    // dispatch a kernel to initialize texture buffers
    initTextures<<<blocks, blockSize>>>(lightmapSize, lightmapDevice, residuesDevice, nextResiduesDevice, sceneDevice);
    checkCUDAError(cudaPeekAtLastError());
    checkCUDAError(cudaDeviceSynchronize());

    u32 kernelQuota = glm::max(1.0, 0.003 * (lightmapSize.x * lightmapSize.y) - 195.608);
    u64 kernelCount = u64(lightmapSize.x * lightmapSize.y) * (lightmapSize.x * lightmapSize.y - 1) / 2 / kernelQuota;
    printf("Kernel quota: %d\n", kernelQuota);
    printf("Kernel count: %lu\n", kernelCount);
    blockSize = 256;
    blocks = ceil(kernelCount / static_cast<f64>(blockSize.x));

    // gather
    for (size_t bounce = 0; bounce < bounces; bounce++) {
        u32 rngSeed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        gather<<<blocks, blockSize>>>(lightmapSize, lightmapDevice, residuesDevice, nextResiduesDevice, sceneDevice, rngSeed, kernelQuota);

        std::swap(residuesDevice, nextResiduesDevice);
        cudaMemsetAsync(nextResiduesDevice, 0, lightmapSize.x * lightmapSize.y * sizeof(vec3));
    }
    checkCUDAError(cudaPeekAtLastError());
    checkCUDAError(cudaDeviceSynchronize());

    // copy lightmap back
    vec3* lightmapHost = new vec3[lightmapSize.x * lightmapSize.y];
    cudaMemcpy(lightmapHost, lightmapDevice, lightmapSize.x * lightmapSize.y * sizeof(vec3), cudaMemcpyDeviceToHost);

    // free everything on the device
    cudaFree(sceneDevice.patches);
    cudaFree(sceneDevice.faces);
    cudaFree(sceneDevice.materials);
    cudaFree(sceneDevice.bvh.nodes);
    cudaFree(lightmapDevice);
    cudaFree(residuesDevice);
    cudaFree(nextResiduesDevice);

    return lightmapHost;
}
