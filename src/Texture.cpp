#include "Texture.h"

#include "RadiositySolver.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

template class Texture<bool>;
template class Texture<float>;
template class Texture<uint64_t>;
template class Texture<vec2>;
template class Texture<vec3>;
template class Texture<vec4>;
template class Texture<Patch>;

template <typename T>
Texture<T>::Texture(const std::filesystem::path& filePath) {
    LOG("Loading image " << filePath);

    int channelsToLoad = 3;
    if constexpr (std::is_same_v<T, float>)
        channelsToLoad = 1;
    else if constexpr (std::is_same_v<T, vec2>)
        channelsToLoad = 2;
    else if constexpr (std::is_same_v<T, vec3>)
        channelsToLoad = 3;
    else if constexpr (std::is_same_v<T, vec4>)
        channelsToLoad = 4;
    else
        assert(false);

    int channels;
    ivec2 sizeInt;

    const auto filePathStr = filePath.string();
    void* imageData;
    imageData = stbi_loadf(filePathStr.c_str(), &sizeInt.x, &sizeInt.y, &channels, channelsToLoad);

    if (!imageData) {
        LOG("Failed to load texture " << filePath);
        throw std::runtime_error("Failed to load texture");
    }

    m_size = sizeInt;
    m_data = reinterpret_cast<T*>(imageData);
    m_stbLoaded = true;
}

template <typename T>
Texture<T>::~Texture() {
    if (!m_data)
		return;

    if (m_stbLoaded)
        stbi_image_free(m_data);
    else
        delete[] m_data;
}

template <typename T>
void Texture<T>::save(const std::filesystem::path& filePath) const {
    LOG("Saving image " << filePath);

    int channelsToSave = 1;
    if constexpr (std::is_same_v<T, float>)
        channelsToSave = 1;
    else if constexpr (std::is_same_v<T, vec2>)
        channelsToSave = 2;
    else if constexpr (std::is_same_v<T, vec3>)
        channelsToSave = 3;
    else if constexpr (std::is_same_v<T, vec4>)
        channelsToSave = 4;
    else
        assert(false);

    const auto filePathStr = filePath.string();
    stbi_flip_vertically_on_write(true);
    stbi_write_hdr(filePathStr.c_str(), m_size.x, m_size.y, channelsToSave, reinterpret_cast<float* const>(m_data));
}
