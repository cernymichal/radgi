#include "BVH.h"

void BVH::build() {
    m_nodes = std::vector<Node>(std::bit_ceil(m_faces.size()) * 2 - 1);

    for (Face& face : m_faces)
        face.calculateAABB();

    buildRecursive(0, static_cast<u32>(m_faces.size()), 0);
}

bool BVH::intersects(const vec3& rayOrigin, const vec3& rayDirection, const Interval<f32>& tInterval, const std::function<bool(f32, const Face&)>& hitPredicate) const {
    vec3 rayDirectionInv = 1.0f / rayDirection;

    u32 stack[64];
    i32 stackSize = 0;
    stack[stackSize++] = 0;

    while (stackSize > 0) {
        auto node = stack[--stackSize];
        auto faceId = m_nodes[node].face;

        if (faceId != u32(-1)) {
            const auto& face = m_faces[faceId];

            if (glm::dot(rayDirection, face.normal) >= 0)
                continue;

            auto t = rayTriangleIntersection(rayOrigin, rayDirection, face.vertices);

            if (!isnan(t) && tInterval.contains(t) && hitPredicate(t, m_faces[faceId]))
                return true;

            continue;
        }

        if (node >= m_nodes.size() / 2)  // Leaf node
            continue;

        auto [tNear, tFar] = rayAABBintersection(rayOrigin, rayDirectionInv, m_nodes[node].aabb);
        if (isnan(tNear) || tNear > tInterval.max || tFar < tInterval.min)
            continue;

        auto leftChild = 2 * node + 1;
        auto rightChild = 2 * node + 2;
        stack[stackSize++] = rightChild;
        stack[stackSize++] = leftChild;

// #define DEBUG_BVH_STACK
#ifdef DEBUG_BVH_STACK
        if (stackSize >= 64) {
            printf("BVH traversal stack overflow\n");
            return false;
        }
#endif
    }

    return false;
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

    auto axis = random<u32>(0, 3);
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
