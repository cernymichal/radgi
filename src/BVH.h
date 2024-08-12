#pragma once

#include "Mesh.h"

constexpr u32 BVH_MAX_DEPTH = 32;
constexpr u32 BVH_MAX_TRIANGLES_PER_LEAF = 16;

class BVH {
public:
    struct Node {
        AABB aabb;
        u32 faceCount;
        union {              // Either faceIndex or childIndex if faceCount == 0
            u32 faceIndex;   // First face
            u32 childIndex;  // Left child
        };
    };

    std::vector<Node> m_nodes;

    BVH(std::vector<Face>& faces) : m_faces(faces) {}

    struct Stats {
        std::chrono::microseconds buildTime;
        u32 faceCount = 0;
        u32 nodeCount = 0;
        u32 leafCount = 0;
        u32 maxDepth = 0;
        u32 maxFacesPerLeaf = 0;
    };

    bool intersect(const vec3& rayOrigin, const vec3& rayDirection, const Interval<f32>& tInterval, const Face* excludedFaces[2]) const;

    void build(u32 perAxisSplitTests = 32);

    bool isBuilt() const { return !m_nodes.empty(); }

    const Stats& stats() const { return m_stats; }

private:
    struct SplitData {
        bool shouldSplit;
        u32 splitAxis;
        f32 splitPoint;
        AABB leftAABB;
        AABB rightAABB;
    };

    std::vector<Face>& m_faces;
    u32 m_perAxisSplitTests = 8;
    Stats m_stats;

    bool splitNode(u32 nodeIndex, std::vector<AABB>& faceAABBs);

    SplitData findBestSplit(u32 nodeIndex, const std::vector<AABB>& faceAABBs) const;
};
