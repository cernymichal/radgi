#pragma once

template <typename T>
class Texture {
public:
    Texture(const uvec2& size) : m_size(size), m_stbLoaded(false) {
        m_data = new T[m_size.x * m_size.y];
    }

    Texture(const std::filesystem::path& filePath);

    ~Texture();

    const T& operator[](const vec2& pos) const {
        return this->operator[](uvec2(pos * vec2(m_size)));
    }

    T& operator[](const vec2& pos) {
        return this->operator[](uvec2(pos * vec2(m_size)));
    }

    const T& operator[](const uvec2& pos) const {
        return m_data[pos.y * m_size.x + pos.x];
    }

    T& operator[](const uvec2& pos) {
        return m_data[pos.y * m_size.x + pos.x];
    }

    void save(const std::filesystem::path& filePath) const;

    void clear(const T& value) {
        for (size_t i = 0; i < m_size.x * m_size.y; i++)
            m_data[i] = value;
    }

    const uvec2& size() const {
        return m_size;
    }

private:
    uvec2 m_size = uvec2(0);
    T* m_data = nullptr;
    bool m_stbLoaded = false;
};
