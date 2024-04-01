#include <math_constants.h>
#include <stdio.h>

#include <chrono>

#include "CUDAStructs.h"
#include "helper_math.h"

using namespace CUDAStructs;

struct Scene {
    Patch* patches;
    Face* faces;
    uint32_t faceCount;
    Material* materials;
};

struct RNG {
    uint32_t state;

    __device__ explicit constexpr RNG(uint32_t seed) : state(seed) {}

    __device__ constexpr inline float operator()(float min = 0.0f, float max = 1.0f) {
        // https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
        // PCG PRNG
        state = state * 747796405u + 2891336453u;
        uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        auto randomValue = (word >> 22u) ^ word;

        // map to a float in [min, max]
        return min + (max - min) * static_cast<float>(randomValue) / static_cast<float>(uint32_t(-1));
    }
};

__device__ float3 randomPointOnPatch(const Patch& patch, RNG& rng) {
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

__device__ float rayTriangleIntersection(const float3& rayOrigin, const float3& rayDirection, const float3 vertices[3]) {
    // X = rayOrigin + rayDirection * t

    // Möller–Trumbore intersection algorithm
    auto edge1 = vertices[1] - vertices[0];
    auto edge2 = vertices[2] - vertices[0];
    auto P = cross(rayDirection, edge2);
    auto determinant = dot(edge1, P);

    // if the determinant is negative, the triangle is back facing
    // if the determinant is close to 0, the ray misses the triangle
    if (determinant < 0.0001f)
        return NAN;

    auto determinantInv = 1.0f / determinant;
    auto T = rayOrigin - vertices[0];
    auto u = dot(T, P) * determinantInv;
    if (u < 0 || u > 1)
        return NAN;

    auto Q = cross(T, edge1);
    auto v = dot(rayDirection, Q) * determinantInv;
    if (v < 0 || u + v > 1)
        return NAN;

    auto t = dot(edge2, Q) * determinantInv;
    if (t < 0)
        return NAN;

    return t;
}

__device__ float calculateFormFactor(const Patch& patchA, const Patch& patchB, const Scene& scene, RNG& rng) {
    float F = 0;

    constexpr auto rayCount = 8;  // TODO make this a parameter
    for (uint32_t i = 0; i < rayCount; i++) {
        auto rayOrigin = randomPointOnPatch(patchA, rng);
        auto rayTarget = randomPointOnPatch(patchB, rng);

        // visibility test
        auto targetDistance = length(rayTarget - rayOrigin);
        auto rayDirection = (rayTarget - rayOrigin) / targetDistance;

        if (dot(rayDirection, scene.faces[patchA.faceId].normal) <= 0 || dot(-rayDirection, scene.faces[patchB.faceId].normal) <= 0)
            continue;

        bool hit = false;
        for (uint32_t i = 0; i < scene.faceCount; i++) {
            auto& face = scene.faces[i];
            if (dot(-rayDirection, face.normal) <= 0)
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

        if (hit)  // visibility test failed
            continue;

        auto ray = rayTarget - rayOrigin;
        auto r2 = dot(ray, ray);
        auto cosines = dot(rayDirection, scene.faces[patchA.faceId].normal) * dot(-rayDirection, scene.faces[patchB.faceId].normal);
        auto deltaF = cosines * patchB.area / (CUDART_PI_F * r2);  // + patchB.area / rayCount);

        if (deltaF > 0)
            F += deltaF;
    }

    return F / rayCount;
}

__global__ void gather(int2 lightmapSize, float3* lightmap, float3* residues, float3* nextResidues, const Scene scene, uint32_t rngSeed) {
    int2 destinationST = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (destinationST.x >= lightmapSize.x || destinationST.y >= lightmapSize.y)
        return;

    int destinationIdx = destinationST.y * lightmapSize.x + destinationST.x;
    auto& destination = scene.patches[destinationIdx];

    if (destination.faceId == NULL_ID)
        return;  // nothing to solve

    RNG rng(rngSeed + destinationIdx);

    // shoot to other patches
    auto shooterST = make_int2(0, 0);
    for (shooterST.y = 0; shooterST.y < lightmapSize.y; shooterST.y++) {
        for (shooterST.x = 0; shooterST.x < lightmapSize.x; shooterST.x++) {
            auto shooterIdx = shooterST.y * lightmapSize.x + shooterST.x;
            auto& shooter = scene.patches[shooterIdx];
            auto shooterResidue = residues[shooterIdx];
            if (shooter.faceId == NULL_ID || shooterIdx == destinationIdx || shooter.faceId == destination.faceId || (shooterResidue.x == 0 && shooterResidue.y == 0 && shooterResidue.z == 0))
                continue;

            // check if the patches are facing each other
            auto sightLine = normalize(shooter.center - destination.center);
            if (dot(sightLine, scene.faces[destination.faceId].normal) <= 0 || dot(-sightLine, scene.faces[shooter.faceId].normal) <= 0)
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

__global__ void initTextures(int2 lightmapSize, float3* lightmap, float3* residues, float3* nextResidues, const Scene scene) {
    int2 texelST = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (texelST.x >= lightmapSize.x || texelST.y >= lightmapSize.y)
        return;

    auto patchIdx = texelST.y * lightmapSize.x + texelST.x;
    auto& patch = scene.patches[patchIdx];

    float3 residue;
    if (patch.faceId != NULL_ID)
        residue = scene.materials[scene.faces[patch.faceId].materialId].emission;
    else
        residue = make_float3(0, 0, 0);

    lightmap[patchIdx] = residue;
    residues[patchIdx] = residue;
    nextResidues[patchIdx] = make_float3(0, 0, 0);
}

__host__ extern "C" float3* solveRadiosityCUDA(uint32_t bounces, int2 lightmapSize, Patch* patches, size_t sceneFaceCount, Face* sceneFaces, size_t materialCount, Material* materials) {
    Face* sceneFacesDevice;
    cudaMalloc(&sceneFacesDevice, sceneFaceCount * sizeof(Face));
    cudaMemcpyAsync(sceneFacesDevice, sceneFaces, sceneFaceCount * sizeof(Face), cudaMemcpyHostToDevice);

    Patch* patchesDevice;
    cudaMalloc(&patchesDevice, lightmapSize.x * lightmapSize.y * sizeof(Patch));
    cudaMemcpyAsync(patchesDevice, patches, lightmapSize.x * lightmapSize.y * sizeof(Patch), cudaMemcpyHostToDevice);

    Material* materialsDevice;
    cudaMalloc(&materialsDevice, materialCount * sizeof(Material));
    cudaMemcpyAsync(materialsDevice, materials, materialCount * sizeof(Material), cudaMemcpyHostToDevice);

    float3* nextResiduesDevice;
    cudaMalloc(&nextResiduesDevice, lightmapSize.x * lightmapSize.y * sizeof(float3));

    float3* lightmapDevice;
    cudaMalloc(&lightmapDevice, lightmapSize.x * lightmapSize.y * sizeof(float3));

    float3* residuesDevice;
    cudaMalloc(&residuesDevice, lightmapSize.x * lightmapSize.y * sizeof(float3));

    dim3 blockSize(16, 16);  // TODO optimize for SM occupancy
    dim3 blocks(ceil(lightmapSize.x / static_cast<double>(blockSize.x)), ceil(lightmapSize.y / static_cast<double>(blockSize.y)));

    Scene scene{patchesDevice, sceneFacesDevice, static_cast<uint32_t>(sceneFaceCount), materialsDevice};

    initTextures<<<blocks, blockSize>>>(lightmapSize, lightmapDevice, residuesDevice, nextResiduesDevice, scene);

    for (size_t bounce = 0; bounce < bounces; bounce++) {
        uint32_t rngSeed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        gather<<<blocks, blockSize>>>(lightmapSize, lightmapDevice, residuesDevice, nextResiduesDevice, scene, rngSeed);

        std::swap(residuesDevice, nextResiduesDevice);
        cudaMemsetAsync(nextResiduesDevice, 0, lightmapSize.x * lightmapSize.y * sizeof(float3));
    }
    cudaDeviceSynchronize();

    float3* lightmapHost = new float3[lightmapSize.x * lightmapSize.y];
    cudaMemcpy(lightmapHost, lightmapDevice, lightmapSize.x * lightmapSize.y * sizeof(float3), cudaMemcpyDeviceToHost);

    cudaFree(patchesDevice);
    cudaFree(sceneFacesDevice);
    cudaFree(materialsDevice);
    cudaFree(lightmapDevice);
    cudaFree(residuesDevice);
    cudaFree(nextResiduesDevice);

    return lightmapHost;
}
