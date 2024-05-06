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

__device__ float calculateFormFactor(const Patch& patchA, const Patch& patchB, const Scene& scene, RNG& rng) {
    float F = 0;

    constexpr auto rayCount = 8;  // TODO make this a parameter
    for (uint32_t i = 0; i < rayCount; i++) {
        auto rayOrigin = randomPointOnPatch(patchA, rng);
        auto rayTarget = randomPointOnPatch(patchB, rng);

        // visibility test
        auto targetDistance = glm::length(rayTarget - rayOrigin);
        auto rayDirection = (rayTarget - rayOrigin) / targetDistance;

        if (glm::dot(rayDirection, scene.faces[patchA.faceId].normal) <= 0 || glm::dot(-rayDirection, scene.faces[patchB.faceId].normal) <= 0)
            continue;

#define USE_BVH
#ifdef USE_BVH
        Interval<float> tInterval = {0, targetDistance - 0.01f};  // leeway for shared edges passing through the lightmap
        uint32_t excludeFaces[] = {patchA.faceId, patchB.faceId};
        bool hit = intersectsBVH(scene, rayOrigin, rayDirection, tInterval, excludeFaces);
#else
        bool hit = false;
        for (uint32_t i = 0; i < scene.faceCount; i++) {
            auto& face = scene.faces[i];
            if (glm::dot(-rayDirection, face.normal) <= 0)
                continue;

            if (i == patchA.faceId || i == patchB.faceId)
                continue;

            auto t = rayTriangleIntersection(rayOrigin, rayDirection, face.vertices);
            if (isnan(t))
                continue;

            if (t < targetDistance - 0.01f) {  // leeway for shared edges passing through the lightmap
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
        float deltaF = cosines * patchB.area / (PI * r2);  // + patchB.area / rayCount);

        if (deltaF > 0)
            F += deltaF;
    }

    return F / rayCount;
}

__global__ void gather(uvec2 lightmapSize, vec3* lightmap, vec3* residues, vec3* nextResidues, const Scene scene, uint32_t rngSeed) {
    uvec2 destinationST = uvec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (destinationST.x >= lightmapSize.x || destinationST.y >= lightmapSize.y)
        return;

    auto destinationIdx = destinationST.y * lightmapSize.x + destinationST.x;
    auto& destination = scene.patches[destinationIdx];

    if (destination.faceId == NULL_ID)
        return;  // nothing to solve

    RNG rng(rngSeed + destinationIdx);

    // shoot to other patches
    auto shooterST = uvec2(0, 0);
    for (shooterST.y = 0; shooterST.y < lightmapSize.y; shooterST.y++) {
        for (shooterST.x = 0; shooterST.x < lightmapSize.x; shooterST.x++) {
            auto shooterIdx = shooterST.y * lightmapSize.x + shooterST.x;
            auto& shooter = scene.patches[shooterIdx];
            auto shooterResidue = residues[shooterIdx];
            if (shooter.faceId == NULL_ID || shooterIdx == destinationIdx || shooter.faceId == destination.faceId || (shooterResidue.x == 0 && shooterResidue.y == 0 && shooterResidue.z == 0))
                continue;

            // check if the patches are facing each other
            auto sightLine = glm::normalize(shooter.vertices[0] - destination.vertices[0]);
            if (glm::dot(sightLine, scene.faces[destination.faceId].normal) <= 0 || glm::dot(-sightLine, scene.faces[shooter.faceId].normal) <= 0)
                continue;

            auto F = calculateFormFactor(shooter, destination, scene, rng);
            if (F == 0)
                continue;

            auto deltaRad = scene.materials[scene.faces[destination.faceId].materialId].albedo * shooterResidue * F * shooter.area / destination.area;
            nextResidues[destinationIdx] += deltaRad;
            lightmap[destinationIdx] += deltaRad;
        }
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

extern "C" vec3* solveRadiosityCUDA(uint32_t bounces, uvec2 lightmapSize, const Scene& sceneHost) {
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
    dim3 blocks(ceil(lightmapSize.x / static_cast<double>(blockSize.x)), ceil(lightmapSize.y / static_cast<double>(blockSize.y)));

    // dispatch a kernel to initialize texture buffers
    initTextures<<<blocks, blockSize>>>(lightmapSize, lightmapDevice, residuesDevice, nextResiduesDevice, sceneDevice);
    checkCUDAError(cudaPeekAtLastError());
    checkCUDAError(cudaDeviceSynchronize());

    // gather
    for (size_t bounce = 0; bounce < bounces; bounce++) {
        uint32_t rngSeed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        gather<<<blocks, blockSize>>>(lightmapSize, lightmapDevice, residuesDevice, nextResiduesDevice, sceneDevice, rngSeed);

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
