#pragma once

#include "Mesh.h"

class BVH {
public:
    BVH(std::vector<Face>& faces) : m_faces(faces) {}

    void build();

    bool intersects(const vec3& rayOrigin, const vec3& rayDirection, const Interval<float>& tInterval, const std::function<bool(float, const Face&)>& hitPredicate) const;

private:
    struct Node {
        AABB aabb;
        uint32_t face = uint32_t(-1);
    };

    std::vector<Face>& m_faces;
    std::vector<Node> m_nodes;

    void buildRecursive(uint32_t startFace, uint32_t endFace, uint32_t node);

    bool intersectsRecursive(const vec3& rayOrigin, const vec3& rayDirection, const vec3& rayDirectionInv, uint32_t node, const Interval<float>& tInterval, const std::function<bool(float, const Face&)>& hitPredicate) const;

    friend class CUDASolver; // needed to make a deep copy of the BVH
};
