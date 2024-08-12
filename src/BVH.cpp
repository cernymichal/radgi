#include "BVH.h"

bool BVH::intersect(const vec3& rayOrigin, const vec3& rayDirection, const Interval<f32>& tInterval, const Face* excludedFaces[2]) const {
    if (m_nodes.empty())
        return false;

    vec3 rayDirectionInv = 1.0f / rayDirection;

    std::array<u32, BVH_MAX_DEPTH> stack;
    u32 stackSize = 0;
    stack[stackSize++] = 0;

    while (stackSize != 0) {
        u32 nodeIndex = stack[--stackSize];
        const Node& node = m_nodes[nodeIndex];

        auto nodeIntersection = rayAABBintersection(rayOrigin, rayDirectionInv, node.aabb);
        if (std::isnan(nodeIntersection.min) || tInterval.intersection(nodeIntersection).length() < 0)
            continue;

        if (node.faceCount != 0) {
            // Leaf node
            for (u32 i = node.faceIndex; i < node.faceIndex + node.faceCount; i++) {
                const Face& face = m_faces.at(i);

                if (&face == excludedFaces[0] || &face == excludedFaces[1])
                    continue;

                auto [t, barycentric] = rayTriangleIntersection(rayOrigin, rayDirection, face.vertices[0], face.vertices[1], face.vertices[2], false);
                if (!std::isnan(t) && tInterval.surrounds(t))
                    return true;
            }

            continue;
        }

        // Add children to stack
        stack[stackSize++] = node.childIndex;
        stack[stackSize++] = node.childIndex + 1;

        // #define DEBUG_BVH_STACK
#ifdef DEBUG_BVH_STACK
        if (stackSize >= BVH_MAX_DEPTH {
            printf("BVH traversal stack overflow\n");
            return false;
        }
#endif
    }

    return false;
}

void BVH::build(u32 perAxisSplitTests) {
    m_perAxisSplitTests = perAxisSplitTests;
    m_stats = Stats();
    m_stats.faceCount = (u32)m_faces.size();
    auto start = std::chrono::high_resolution_clock::now();

    m_nodes.clear();
    m_nodes.reserve(std::bit_ceil(m_faces.size() / BVH_MAX_TRIANGLES_PER_LEAF + 1) * 2 - 1);  // leaftCount * 2 - 1

    // Pre-calculate AABBs for each face
    std::vector<AABB> faceAABBs;
    faceAABBs.reserve(m_faces.size());
    for (const Face& face : m_faces) {
        AABB aabb = AABB::empty();
        for (u32 i = 0; i < 3; i++)
            aabb = aabb.extendTo(face.vertices[i]);
        faceAABBs.push_back(aabb);
    }

    // Create root node
    Node rootNode = {
        .aabb = AABB::empty(),
        .faceCount = (u32)m_faces.size(),
        .faceIndex = 0,
    };
    for (u32 i = rootNode.faceIndex; i < rootNode.faceIndex + rootNode.faceCount; i++)
        rootNode.aabb = rootNode.aabb.boundingUnion(faceAABBs[i]);

    m_nodes.push_back(rootNode);

    std::queue<std::pair<u32, u32>> queue;
    queue.push({0, 1});

    while (!queue.empty()) {
        u32 currentNodeIndex = queue.front().first;
        u32 depth = queue.front().second;
        m_stats.maxDepth = std::max(m_stats.maxDepth, depth);
        queue.pop();

        // Split Node
        bool shouldSplit = m_nodes[currentNodeIndex].faceCount > BVH_MAX_TRIANGLES_PER_LEAF && depth < BVH_MAX_DEPTH;
        bool splitSuccess = shouldSplit ? splitNode(currentNodeIndex, faceAABBs) : false;
        if (!splitSuccess) {
            m_stats.leafCount++;
            m_stats.maxFacesPerLeaf = std::max(m_stats.maxFacesPerLeaf, m_nodes[currentNodeIndex].faceCount);
            continue;
        }

        // Add children to queue
        queue.push({m_nodes[currentNodeIndex].childIndex, depth + 1});
        queue.push({m_nodes[currentNodeIndex].childIndex + 1, depth + 1});
    }

    m_nodes.shrink_to_fit();

    auto end = std::chrono::high_resolution_clock::now();
    m_stats.buildTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    m_stats.nodeCount = (u32)m_nodes.size();
}

bool BVH::splitNode(u32 nodeIndex, std::vector<AABB>& faceAABBs) {
    Node& parentNode = m_nodes[nodeIndex];

    // Split faces
    BVH::SplitData split = findBestSplit(nodeIndex, faceAABBs);
    if (!split.shouldSplit)
        return false;

    // Sort faces based on split
    u32 j = parentNode.faceIndex + parentNode.faceCount - 1;
    for (u32 i = parentNode.faceIndex; i <= j;) {
        if (faceAABBs[i].center()[split.splitAxis] < split.splitPoint)
            i++;
        else {
            std::swap(m_faces[i], m_faces[j]);
            std::swap(faceAABBs[i], faceAABBs[j]);
            j--;
        }
    }

    // Split node
    Node leftChild = {
        .aabb = split.leftAABB,
        .faceCount = j - parentNode.faceIndex + 1,
        .faceIndex = parentNode.faceIndex,
    };

    Node rightChild = {
        .aabb = split.rightAABB,
        .faceCount = parentNode.faceCount - leftChild.faceCount,
        .faceIndex = leftChild.faceIndex + leftChild.faceCount,
    };

    parentNode.faceCount = 0;
    parentNode.childIndex = (u32)m_nodes.size();

    // BEWARE this could invalidate node references on resize!
    m_nodes.push_back(leftChild);
    m_nodes.push_back(rightChild);
    return true;
}

BVH::SplitData BVH::findBestSplit(u32 nodeIndex, const std::vector<AABB>& faceAABBs) const {
    const Node& parentNode = m_nodes[nodeIndex];
    f32 parentCost = parentNode.faceCount * AABBSurfaceArea(parentNode.aabb);  // Surface Area Heuristic

    f32 bestCost = parentCost;
    BVH::SplitData bestSplitData = {false, 0, 0.0f, AABB::empty(), AABB::empty()};

    for (u32 splitAxis = 0; splitAxis < 3; splitAxis++) {
        // Calculate real extents of the split axis from centers
        f32 axisStart = parentNode.aabb.min[splitAxis];
        f32 axisEnd = parentNode.aabb.max[splitAxis];
        for (u32 i = parentNode.faceIndex; i < parentNode.faceIndex + parentNode.faceCount; i++) {
            axisStart = std::min(axisStart, faceAABBs[i].center()[splitAxis]);
            axisEnd = std::max(axisEnd, faceAABBs[i].center()[splitAxis]);
        }

        // Initialize bins
        f32 testInterval = (axisEnd - axisStart) / (m_perAxisSplitTests + 1);
        std::vector<std::pair<AABB, u32>> intervalBins(m_perAxisSplitTests + 1, {AABB::empty(), 0});

        // Bin faces
        for (u32 i = parentNode.faceIndex; i < parentNode.faceIndex + parentNode.faceCount; i++) {
            u32 binIndex = floor((faceAABBs[i].center()[splitAxis] - axisStart) / testInterval);
            binIndex = glm::clamp(binIndex, 0U, m_perAxisSplitTests);
            intervalBins[binIndex].first = intervalBins[binIndex].first.boundingUnion(faceAABBs[i]);
            intervalBins[binIndex].second++;
        }

        for (u32 splitNum = 0; splitNum < m_perAxisSplitTests; splitNum++) {
            SplitData split = {
                .shouldSplit = true,
                .splitAxis = splitAxis,
                .splitPoint = axisStart + testInterval * (splitNum + 1),
                .leftAABB = AABB::empty(),
                .rightAABB = AABB::empty(),
            };
            u32 leftCount = 0;

            // Sum up left and right bins
            for (u32 i = 0; i <= splitNum; i++) {
                split.leftAABB = split.leftAABB.boundingUnion(intervalBins[i].first);
                leftCount += intervalBins[i].second;
            }
            for (u32 i = splitNum + 1; i < intervalBins.size(); i++)
                split.rightAABB = split.rightAABB.boundingUnion(intervalBins[i].first);

            if (leftCount == 0 || leftCount == parentNode.faceCount)
                continue;

            // Calculate SAH cost
            f32 leftCost = leftCount * AABBSurfaceArea(split.leftAABB);
            f32 rightCost = (parentNode.faceCount - leftCount) * AABBSurfaceArea(split.rightAABB);
            f32 cost = leftCost + rightCost;

            if (cost < bestCost) {
                bestCost = cost;
                bestSplitData = split;
            }
        }
    }

    return bestSplitData;
}
