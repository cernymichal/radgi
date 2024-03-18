#include "Texture.h"

#include "Scene.h"

#define TINYEXR_IMPLEMENTATION
#define TINYEXR_USE_MINIZ 0
#include <miniz/miniz.h>
#define TINYEXR_USE_THREAD 1
#include <tinyexr.h>

template class Texture<bool>;
template class Texture<float>;
template class Texture<uint64_t>;
template class Texture<vec2>;
template class Texture<vec3>;
template class Texture<vec4>;
template class Texture<Patch>;

template <typename T>
Texture<T>::Texture(const std::filesystem::path& filePath, bool flipVertically) {
    std::string pathString = filePath.string();
    LOG("Loading texture " << pathString);

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

    EXRVersion version;
    int status = ParseEXRVersionFromFile(&version, pathString.c_str());
    if (status != TINYEXR_SUCCESS || version.multipart) {
        LOG("Invalid EXR file");
        throw std::runtime_error("Invalid EXR file");
    }

    EXRHeader header;
    InitEXRHeader(&header);
    const char* error = nullptr;
    status = ParseEXRHeaderFromFile(&header, &version, pathString.c_str(), &error);

    if (status == TINYEXR_SUCCESS && header.num_channels < channelsToLoad) {
        error = "Not enough channels";
        status = -1;
    }

    if (status != TINYEXR_SUCCESS) {
        LOG("Invalid EXR file");
        LOG(error);
        FreeEXRErrorMessage(error);
        throw std::runtime_error("Invalid EXR file");
    }

    // Read HALF channel as FLOAT.
    for (int i = 0; i < header.num_channels; i++) {
        if (header.pixel_types[i] == TINYEXR_PIXELTYPE_HALF)
            header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
    }

    EXRImage image;
    InitEXRImage(&image);
    status = LoadEXRImageFromFile(&image, &header, pathString.c_str(), &error);
    if (status != 0) {
        LOG("Couldn't load EXR file");
        LOG(error);
        FreeEXRHeader(&header);
        FreeEXRErrorMessage(error);
        throw std::runtime_error("Couldn't load EXR file");
    }

    m_size = ivec2(image.width, image.height);
    m_data = new T[m_size.x * m_size.y];

    // Load the image data to a single array, flipping it vertically if necessary
    float* dataFloat = reinterpret_cast<float*>(m_data);
    for (int channel = 0; channel < channelsToLoad; channel++) {
        auto idx = uvec2(0);
        for (idx.y = 0; idx.y < m_size.y; idx.y++) {
            for (idx.x = 0; idx.x < m_size.x; idx.x++) {
                auto i = idx.y * m_size.x + idx.x;
                auto flippedI = (flipVertically ? m_size.y - 1 - idx.y : idx.y) * m_size.x + idx.x;
                dataFloat[flippedI * channelsToLoad + channel] = reinterpret_cast<float*>(image.images[channel])[i];
            }
        }
    }

    // Swap R and B channels because of BGRA format
    if (channelsToLoad >= 3) {
        auto idx = uvec2(0);
        for (idx.y = 0; idx.y < m_size.y; idx.y++) {
            for (idx.x = 0; idx.x < m_size.x; idx.x++) {
                auto i = (idx.y * m_size.x + idx.x) * channelsToLoad;
                std::swap(dataFloat[i], dataFloat[i + 2]);
            }
        }
    }

    FreeEXRImage(&image);
    FreeEXRHeader(&header);
}

template <typename T>
void Texture<T>::save(const std::filesystem::path& filePath, bool flipVertically) const {
    std::string pathString = filePath.string();
    LOG("Saving texture " << pathString);

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

    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    std::vector<float*> images;
    for (int i = 0; i < channelsToSave; i++)
        images.push_back(new float[m_size.x * m_size.y]);

    float* const dataFloat = reinterpret_cast<float* const>(m_data);
    auto idx = uvec2(0);
    for (idx.y = 0; idx.y < m_size.y; idx.y++) {
        for (idx.x = 0; idx.x < m_size.x; idx.x++) {
            auto i = idx.y * m_size.x + idx.x;
            auto flippedI = (flipVertically ? m_size.y - 1 - idx.y : idx.y) * m_size.x + idx.x;
            for (int channel = 0; channel < channelsToSave; channel++)
                images[channel][flippedI] = dataFloat[i * channelsToSave + channel];
        }
    }

    // Must be BGR(A) order, since most of EXR viewers expect this channel order.
    if (channelsToSave >= 3)
        std::swap(images[0], images[2]);

    image.images = reinterpret_cast<unsigned char**>(images.data());
    image.width = m_size.x;
    image.height = m_size.y;
    image.num_channels = channelsToSave;

    std::vector<EXRChannelInfo> channels(channelsToSave, EXRChannelInfo());
    const char* channelNames[] = {"B", "G", "R", "A"};
    for (int i = 0; i < channelsToSave; i++) {
        auto nameLength = strlen(channelNames[i]);
        for (int j = 0; j < nameLength; j++)
            channels[i].name[j] = channelNames[i][j];
        channels[i].name[nameLength] = '\0';
    }
    if (channelsToSave < 3)
        header.channels[0].name[0] = 'R';

    std::vector<int> pixelTypes(channelsToSave, TINYEXR_PIXELTYPE_FLOAT);
    std::vector<int> requestedPixelTypes(channelsToSave, TINYEXR_PIXELTYPE_FLOAT);  // could be HALF

    header.channels = channels.data();
    header.num_channels = image.num_channels;
    header.pixel_types = pixelTypes.data();
    header.requested_pixel_types = requestedPixelTypes.data();
    header.compression_type = TINYEXR_COMPRESSIONTYPE_ZIP;  // TINYEXR_COMPRESSIONTYPE_PIZ;

    const char* error;
    int status = SaveEXRImageToFile(&image, &header, pathString.c_str(), &error);

    for (int i = 0; i < images.size(); i++)
        delete[] images[i];

    if (status != TINYEXR_SUCCESS) {
        LOG("Saving texture failed");
        LOG(error);
        FreeEXRErrorMessage(error);
        throw std::runtime_error("Saving texture failed");
    }
}
