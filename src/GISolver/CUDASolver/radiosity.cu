#include <stdio.h>

#include <chrono>

#include "CUDAStructs.h"
#include "utils.cuh"

using namespace CUDAStructs;

__device__ vec3 randomPointOnPatch(const Patch& patch, RNG& rng) {
    auto u = rng();
    if (patch.vertexCount == 3) {
        auto v = rng(0, 1.0f - u);
        auto w = 1.0f - u - v;
        return u * patch.vertices[0] + v * patch.vertices[1] + w * patch.vertices[2];
    }

    auto edge0 = patch.vertices[1] - patch.vertices[0];
    auto edge1 = patch.vertices[3] - patch.vertices[0];
    return patch.vertices[0] + u * edge0 + rng() * edge1;  // TODO this is a little wrong
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

        f32 r2 = glm::length2(rayTarget - rayOrigin);
        f32 cosines = glm::dot(rayDirection, scene.faces[patchA.faceId].normal) * glm::dot(-rayDirection, scene.faces[patchB.faceId].normal);
        f32 deltaF = cosines / (pi_v<f32> * r2);
        F += glm::max(deltaF, 0.0f);
    }

    return F / rayCount;
}

__device__ __forceinline__ void atomicAdd(hvec3* target, const hvec3& value) {
    atomicAdd(&target->x, value.x);
    atomicAdd(&target->y, value.y);
    atomicAdd(&target->z, value.z);
}

__global__ void gatherPair(const uvec2 lightmapSize, hvec3* lightmap, const hvec3* residues, hvec3* nextResidues, const Scene scene, const u32 rngSeed, u32 threadQuota) {
    u64 threadId = blockIdx.x * blockDim.x + threadIdx.x;
    u64 patchCount = lightmapSize.x * lightmapSize.y;

    glm::vec<2, u64> patchPair;
    {
        const u64 startCombinationIdx = threadId * threadQuota;
        patchPair.y = floor((sqrt(8.0 * startCombinationIdx + 1) + 1) / 2);
        const auto prev_y = patchPair.y - 1;
        patchPair.x = startCombinationIdx - prev_y * (prev_y - 1) / 2 - prev_y;
    }

    RNG rng(rngSeed + threadId);

    for (i32 i = 0; i < threadQuota; i++) {
        if (i != 0)
            patchPair.x++;
        if (patchPair.x >= patchPair.y) {
            patchPair.x = 0;
            patchPair.y++;
        }

        if (patchPair.x >= patchCount || patchPair.y >= patchCount)
            return;

        auto& patchA = scene.patches[patchPair.x];
        auto& patchB = scene.patches[patchPair.y];
        if (patchA.faceId == NULL_ID || patchB.faceId == NULL_ID || patchA.faceId == patchB.faceId)
            continue;

        hvec3 patchAResidue = residues[patchPair.x];
        hvec3 patchBResidue = residues[patchPair.y];
        if (patchAResidue == hvec3(0) && patchBResidue == hvec3(0))
            continue;

        // check if the patches are facing each other
        auto sightLine = glm::normalize(patchB.vertices[0] - patchA.vertices[0]);
        if (glm::dot(sightLine, scene.faces[patchA.faceId].normal) <= 0 || glm::dot(-sightLine, scene.faces[patchB.faceId].normal) <= 0)
            continue;

        f16 F = calculateFormFactor(patchA, patchB, scene, rng);
        if (F <= static_cast<f16>(0.0001f))
            continue;

        auto deltaRadA = scene.materials[scene.faces[patchA.faceId].materialId].albedo * patchBResidue * F * patchB.area;
        auto deltaRadB = scene.materials[scene.faces[patchB.faceId].materialId].albedo * patchAResidue * F * patchA.area;

        atomicAdd(&nextResidues[patchPair.x], deltaRadA);
        atomicAdd(&lightmap[patchPair.x], deltaRadA);
        atomicAdd(&nextResidues[patchPair.y], deltaRadB);
        atomicAdd(&lightmap[patchPair.y], deltaRadB);
    }
}

__global__ void gatherWave(const uvec2 lightmapSize, hvec3* lightmap, const hvec3* residues, hvec3* nextResidues, const Scene scene, const u32 rngSeed, u32 wave) {
    u64 threadId = blockIdx.x * blockDim.x + threadIdx.x;
    u64 patchCount = lightmapSize.x * lightmapSize.y;

    glm::vec<2, u64> patchPair = {threadId, threadId + wave + 1};
    if (patchPair.x >= patchCount || patchPair.y >= patchCount)
        return;

    RNG rng(rngSeed + threadId);

    auto& patchA = scene.patches[patchPair.x];
    auto& patchB = scene.patches[patchPair.y];
    if (patchA.faceId == NULL_ID || patchB.faceId == NULL_ID || patchA.faceId == patchB.faceId)
        return;

    hvec3 patchAResidue = residues[patchPair.x];
    hvec3 patchBResidue = residues[patchPair.y];
    if (patchAResidue == hvec3(0) && patchBResidue == hvec3(0))
        return;

    // check if the patches are facing each other
    auto sightLine = glm::normalize(patchB.vertices[0] - patchA.vertices[0]);
    if (glm::dot(sightLine, scene.faces[patchA.faceId].normal) <= 0 || glm::dot(-sightLine, scene.faces[patchB.faceId].normal) <= 0)
        return;

    f16 F = calculateFormFactor(patchA, patchB, scene, rng);
    if (F <= static_cast<f16>(0.0001f))
        return;

    auto deltaRadA = scene.materials[scene.faces[patchA.faceId].materialId].albedo * patchBResidue * F * patchB.area;
    auto deltaRadB = scene.materials[scene.faces[patchB.faceId].materialId].albedo * patchAResidue * F * patchA.area;

    nextResidues[patchPair.x] += deltaRadA;
    lightmap[patchPair.x] += deltaRadA;
    nextResidues[patchPair.y] += deltaRadB;
    lightmap[patchPair.y] += deltaRadB;
}

__global__ void gather(const uvec2 lightmapSize, hvec3* lightmap, const hvec3* residues, hvec3* nextResidues, const Scene scene, const u32 rngSeed) {
    uvec2 destinationST = uvec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (glm::any(destinationST >= lightmapSize))
        return;

    // TODO remove xy coordinates

    auto destinationIdx = destinationST.y * lightmapSize.x + destinationST.x;
    auto& destination = scene.patches[destinationIdx];

    if (destination.faceId == NULL_ID)
        return;  // nothing to solve

    RNG rng(rngSeed + destinationIdx);

    hvec3 deltaRad = hvec3(0);

    auto shooterST = uvec2(0, 0);
    for (shooterST.y = 0; shooterST.y < lightmapSize.y; shooterST.y++) {
        for (shooterST.x = 0; shooterST.x < lightmapSize.x; shooterST.x++) {
            auto shooterIdx = shooterST.y * lightmapSize.x + shooterST.x;
            auto& shooter = scene.patches[shooterIdx];
            auto shooterResidue = residues[shooterIdx];
            if (shooter.faceId == NULL_ID || shooterIdx == destinationIdx || shooter.faceId == destination.faceId || shooterResidue == hvec3(0))
                continue;

            // check if the patches are facing each other
            auto sightLine = glm::normalize(shooter.vertices[0] - destination.vertices[0]);
            if (glm::dot(sightLine, scene.faces[destination.faceId].normal) <= 0 || glm::dot(-sightLine, scene.faces[shooter.faceId].normal) <= 0)
                continue;

            f16 F = calculateFormFactor(shooter, destination, scene, rng);
            if (F <= static_cast<f16>(0.0001f))
                continue;

            deltaRad += shooterResidue * F * shooter.area;
        }
    }

    deltaRad *= scene.materials[scene.faces[destination.faceId].materialId].albedo;
    nextResidues[destinationIdx] += deltaRad;
    lightmap[destinationIdx] += deltaRad;
}

__global__ void initTextures(uvec2 lightmapSize, hvec3* lightmap, hvec3* residues, hvec3* nextResidues, const Scene scene) {
    auto texelST = uvec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (texelST.x >= lightmapSize.x || texelST.y >= lightmapSize.y)
        return;

    auto patchIdx = texelST.y * lightmapSize.x + texelST.x;
    auto& patch = scene.patches[patchIdx];

    hvec3 residue;
    if (patch.faceId != NULL_ID)
        residue = scene.materials[scene.faces[patch.faceId].materialId].emission;
    else
        residue = hvec3(0);

    lightmap[patchIdx] = residue;
    residues[patchIdx] = residue;
    nextResidues[patchIdx] = hvec3(0);
}

extern "C" hvec3* solveRadiosityCUDA(u32 bounces, uvec2 lightmapSize, const Scene& sceneHost) {
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
    hvec3* lightmapDevice;
    cudaMalloc(&lightmapDevice, lightmapSize.x * lightmapSize.y * sizeof(hvec3));

    hvec3* residuesDevice;
    cudaMalloc(&residuesDevice, lightmapSize.x * lightmapSize.y * sizeof(hvec3));

    hvec3* nextResiduesDevice;
    cudaMalloc(&nextResiduesDevice, lightmapSize.x * lightmapSize.y * sizeof(hvec3));

    checkCUDAError(cudaDeviceSynchronize());
    checkCUDAError(cudaPeekAtLastError());

    dim3 blockSize(16, 16);  // TODO optimize for SM occupancy
    dim3 blocks((u32)ceil(lightmapSize.x / static_cast<f64>(blockSize.x)), (u32)ceil(lightmapSize.y / static_cast<f64>(blockSize.y)));

    // dispatch a kernel to initialize texture buffers
    initTextures<<<blocks, blockSize>>>(lightmapSize, lightmapDevice, residuesDevice, nextResiduesDevice, sceneDevice);
    checkCUDAError(cudaDeviceSynchronize());
    checkCUDAError(cudaPeekAtLastError());

#if defined(USE_PAIR_GATHER)

    u32 threadQuota = static_cast<u32>(glm::max(1.0, 0.003 * (lightmapSize.x * lightmapSize.y) - 195.608));  // magic function to determine thread quota
    u64 kernelCount = u64(lightmapSize.x * lightmapSize.y) * (lightmapSize.x * lightmapSize.y - 1) / 2 / threadQuota;
    printf("Thread quota: %d\n", threadQuota);
    printf("Thread count: %llu\n", kernelCount);
    blockSize = 256;
    blocks = static_cast<u32>(ceil(kernelCount / static_cast<f64>(blockSize.x)));

    // gather
    for (size_t bounce = 0; bounce < bounces; bounce++) {
        u32 rngSeed = static_cast<u32>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
        gatherPair<<<blocks, blockSize>>>(lightmapSize, lightmapDevice, residuesDevice, nextResiduesDevice, sceneDevice, rngSeed, threadQuota);

        std::swap(residuesDevice, nextResiduesDevice);
        cudaMemset(nextResiduesDevice, 0, lightmapSize.x * lightmapSize.y * sizeof(hvec3));
    }

#elif defined(USE_WAVE_GATHER)

    u32 patchCount = lightmapSize.x * lightmapSize.y;
    blockSize = 256;

    // TODO if too little blocks, use a different kernel

    // gather
    for (size_t bounce = 0; bounce < bounces; bounce++) {
        u32 rngSeed = static_cast<u32>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
        for (u32 wave = 0; patchCount - wave - 1 >= 1; wave++) {
            blocks = static_cast<u32>(ceil((patchCount - wave - 1) / static_cast<f64>(blockSize.x)));
            gatherWave<<<blocks, blockSize>>>(lightmapSize, lightmapDevice, residuesDevice, nextResiduesDevice, sceneDevice, rngSeed, wave);
        }

        std::swap(residuesDevice, nextResiduesDevice);
        cudaMemset(nextResiduesDevice, 0, lightmapSize.x * lightmapSize.y * sizeof(hvec3));
    }

#else  // simple per patch gather

    // gather
    for (size_t bounce = 0; bounce < bounces; bounce++) {
        u32 rngSeed = static_cast<u32>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
        gather<<<blocks, blockSize>>>(lightmapSize, lightmapDevice, residuesDevice, nextResiduesDevice, sceneDevice, rngSeed);

        std::swap(residuesDevice, nextResiduesDevice);
        cudaMemset(nextResiduesDevice, 0, lightmapSize.x * lightmapSize.y * sizeof(hvec3));
    }

#endif

    checkCUDAError(cudaDeviceSynchronize());
    checkCUDAError(cudaPeekAtLastError());

    // copy lightmap back
    hvec3* lightmapHost = new hvec3[lightmapSize.x * lightmapSize.y];
    cudaMemcpy(lightmapHost, lightmapDevice, lightmapSize.x * lightmapSize.y * sizeof(hvec3), cudaMemcpyDeviceToHost);

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
