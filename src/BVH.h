#pragma once

#include "Mesh.h"

class BVH {
public:
    BVH(std::vector<Face>& faces) : m_faces(faces) {}

    void build();

    bool intersects(const vec3& rayOrigin, const vec3& rayDirection, const Interval<f32>& tInterval, const std::function<bool(f32, const Face&)>& hitPredicate) const;

private:
    struct Node {
        AABB aabb;
        u32 face = u32(-1);
    };

    std::vector<Face>& m_faces;
    std::vector<Node> m_nodes;

    void buildRecursive(u32 startFace, u32 endFace, u32 node);

    friend class CUDASolver;  // needed to make a deep copy of the BVH
};
