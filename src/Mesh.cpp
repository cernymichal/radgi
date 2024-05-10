#include "Mesh.h"

#include <fstream>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

std::vector<Face> loadMesh(const std::filesystem::path& filePath) {
    LOG("Loading mesh " << filePath);

    tinyobj::ObjReaderConfig reader_config;
    reader_config.triangulate = true;

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filePath.string(), reader_config)) {
        if (!reader.Error().empty())
            LOG("tinyobjloader: " << reader.Error());

        throw std::runtime_error("Failed to load mesh");
    }

    if (!reader.Warning().empty())
        LOG("tinyobjloader: " << reader.Warning());

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();
    std::vector<Ref<Material>> materialRefs;

    for (const auto& material : materials) {
        auto mat = makeRef<Material>();
        mat->albedo = std::bit_cast<vec3>(material.diffuse);
        mat->emission = std::bit_cast<vec3>(material.emission);
        materialRefs.push_back(mat);
    }

    std::vector<Face> faces;

    for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            auto& face = faces.emplace_back();
            face.material = materialRefs[shapes[s].mesh.material_ids[f]];

            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
            assert(fv == 3);
            for (size_t v = 0; v < fv; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
                face.vertices[v] = vec3(vx, vy, vz);

                assert(idx.normal_index >= 0);
                tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                face.normal = vec3(nx, ny, nz);

                assert(idx.texcoord_index >= 0);
                tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                face.lightmapUVs[v] = vec2(tx, ty);
            }
            // face.normal = glm::normalize(glm::cross(face.vertices[1] - face.vertices[0], face.vertices[2] - face.vertices[0]));

            index_offset += fv;
        }
    }

    return faces;
}

void saveMesh(const std::filesystem::path& filePath, const std::vector<Face>& faces) {
    auto file = std::ofstream(filePath);

    auto patchIdx = uvec2(0, 0);
    auto vertexI = 1;
    for (auto& face : faces) {
        for (u32 i = 0; i < 3; i++) {
            auto vertex = face.vertices[i];
            file << std::format("v {} {} {}\n", vertex.x, vertex.y, vertex.z);
            vertexI++;
        }

        file << std::format("f {} {} {}\n", vertexI - 3, vertexI - 2, vertexI - 1);
    }

    file.close();
}
