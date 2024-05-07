#include "BVH.h"

void BVH::build() {
    m_nodes = std::vector<Node>(std::bit_ceil(m_faces.size()) * 2 - 1);

    for (Face& face : m_faces)
        face.aabb = face.calculateAABB();

    buildRecursive(0, m_faces.size(), 0);
}

bool BVH::intersects(const vec3& rayOrigin, const vec3& rayDirection, const Interval<f32>& tInterval, const std::function<bool(f32, const Face&)>& hitPredicate) const {
    return intersectsRecursive(rayOrigin, rayDirection, 1.0f / rayDirection, 0, tInterval, hitPredicate);
}

void BVH::buildRecursive(u32 startFace, u32 endFace, u32 node) {
    if (endFace - startFace == 0) {
        m_nodes[node].face = u32(-1);
        m_nodes[node].aabb = AABB::empty();
        return;
    }

    if (endFace - startFace == 1) {
        m_nodes[node].face = startFace;
        m_nodes[node].aabb = m_faces[startFace].aabb;
        return;
    }

    auto axis = random<u32>(0, 2);
    std::sort(m_faces.begin() + startFace, m_faces.begin() + endFace, [axis](const Face& a, const Face& b) {
        return a.aabb.min[axis] < b.aabb.min[axis];
    });

    u32 median = (startFace + endFace) / 2;
    auto leftChild = 2 * node + 1;
    auto rightChild = 2 * node + 2;
    buildRecursive(startFace, median, leftChild);
    buildRecursive(median, endFace, rightChild);

    m_nodes[node].aabb = m_nodes[leftChild].aabb.boundingUnion(m_nodes[rightChild].aabb);
}

bool BVH::intersectsRecursive(const vec3& rayOrigin, const vec3& rayDirection, const vec3& rayDirectionInv, u32 node, const Interval<f32>& tInterval, const std::function<bool(f32, const Face&)>& hitPredicate) const {
    if (node >= m_nodes.size() / 2) {
        // Leaf node

        if (m_nodes[node].face == u32(-1))
            return false;

        const auto& face = m_faces[m_nodes[node].face];

        if (glm::dot(rayDirection, face.normal) >= 0)
            return false;

        auto t = rayTriangleIntersection(rayOrigin, rayDirection, face.vertices);
        return !std::isnan(t) && tInterval.contains(t) && hitPredicate(t, face);
    }

    auto [tNear, tFar] = rayAABBintersection(rayOrigin, rayDirectionInv, m_nodes[node].aabb);
    if (std::isnan(tNear) || tNear > tInterval.max || tFar < tInterval.min)
        return false;

    auto leftChild = 2 * node + 1;
    auto rightChild = 2 * node + 2;
    return intersectsRecursive(rayOrigin, rayDirection, rayDirectionInv, leftChild, tInterval, hitPredicate) ||
           intersectsRecursive(rayOrigin, rayDirection, rayDirectionInv, rightChild, tInterval, hitPredicate);
}
