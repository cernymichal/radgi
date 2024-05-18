#pragma once

#include <chrono>
#include <random>
#include <thread>

#include "Math.h"
#include "Scalars.h"

/*
 * @brief Fast pseudo random number generator by Sebastiano Vigna
 */
struct SplitMix64 {
    u64 state;

    /*
     * @brief Constructs a new instance.
     * @param seed Initial state.
     */
    explicit constexpr SplitMix64(u64 seed) : state(seed) {}

    /*
     * @brief Constructs a new instance seeded by the current time.
     */
    explicit SplitMix64() : SplitMix64(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count()) {}

    /*
     * @brief Generates a random uint64 and updates the state.
     */
    constexpr inline u64 operator()() {
        // https://prng.di.unimi.it/splitmix64.c

        state += 0x9e3779b97f4a7c15Ui64;

        u64 z = state;
        z ^= z >> 30;
        z *= 0xbf58476d1ce4e5b9Ui64;
        z ^= z >> 27;
        z *= 0x94d049bb133111ebUi64;
        z ^= z >> 31;

        return z;
    }

    static inline constexpr u64 min() {
        return 0;
    }

    static inline constexpr u64 max() {
        return u64(-1);
    }
};

/*
 * @brief Very fast solid pseudo random number generator by David Blackman and Sebastiano Vigna
 *
 * https://prng.di.unimi.it/
 */
struct Xoshiro256SS {
    u64 state[4];

    /*
     * @brief Constructs a new instance.
     * @param seed 64b seed from which the initial state is generated using splitmix64.
     */
    explicit constexpr Xoshiro256SS(u64 seed) {
        SplitMix64 generator(seed);

        state[0] = generator();
        state[1] = generator();
        state[2] = generator();
        state[3] = generator();
    }

    /*
     * @brief Constructs a new instance seeded by the current time and thread ID.
     */
    explicit Xoshiro256SS() : Xoshiro256SS(std::chrono::system_clock::now().time_since_epoch().count() + std::hash<std::thread::id>()(std::this_thread::get_id())) {}

    /*
     * @brief Generates a random uint64 and updates the state.
     */
    constexpr inline u64 operator()() {
        // https://prng.di.unimi.it/xoshiro256starstar.c

        u64 result = bitRotateLeft(state[1] * 5, 7) * 9;

        u64 t = state[1] << 17;
        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];
        state[2] ^= t;
        state[3] = bitRotateLeft(state[3], 45);

        return result;
    }

    static inline constexpr u64 min() {
        return 0;
    }

    static inline constexpr u64 max() {
        return u64(-1);
    }

private:
    /*
     * @return value rotated by k places left.
     */
    static inline constexpr u64 bitRotateLeft(u64 value, i32 k) {
        return (value << k) | (value >> (64 - k));
    }
};

inline thread_local Xoshiro256SS RANDOM_GENERATOR;

/*
 * @param min The minimum value of the range.
 * @param max The maximum value of the range.
 * @return A random T in the range [min, max).
 *
 * @note Uses RANDOM_GENERATOR internally.
 */
template <typename T>
inline T random(T min, T max) {
    if constexpr (std::is_floating_point_v<T>)
        return min + (max - min) * static_cast<T>(RANDOM_GENERATOR()) / static_cast<T>(RANDOM_GENERATOR.max());
    else
        return min + RANDOM_GENERATOR() % (max - min);
}

/*
 * @return A random value of T or a floating-point number in the range [0, 1).
 *
 * @note Uses RANDOM_GENERATOR internally.
 */
template <typename T>
inline T random() {
    if constexpr (std::is_floating_point_v<T>)
        return random<T>(0, 1);
    else if constexpr (std::is_same_v<T, bool>)
        return RANDOM_GENERATOR() & 1;
    else {
        auto value = RANDOM_GENERATOR();
        return std::bit_cast<T>(value);
    }
}

/*
 * @return A random value from a normal distribution N(0, 1).
 *
 * @note Uses RANDOM_GENERATOR internally.
 */
template <typename T>
inline T randomNormal() {
    // https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    // faster than ratio of uniforms :( but ziggurat might be faster

    static_assert(std::is_floating_point_v<T>, "Only floating-point types are supported.");

    thread_local std::pair<T, bool> cache(0, false);  // {cached value, valid}
    if (cache.second) {
        cache.second = false;
        return cache.first;
    }

    T u1, u2;
    do {
        u1 = random<T>();
    } while (u1 == 0);
    u2 = random<T>();

    T mag = 1 * sqrt(-2 * log(u1));
    T z0 = mag * cos(2 * pi_v<T> * u2);
    T z1 = mag * sin(2 * pi_v<T> * u2);

    cache = {z1, true};
    return z0;
}

// Random vectors

/*
 * @return A random vector with values in the range in the range [min, max).
 *
 * @note Uses RANDOM_GENERATOR internally.
 */
template <int L, typename T = f32>
inline glm::vec<L, T> randomVec(const glm::vec<L, T>& min, const glm::vec<L, T>& max) {
    glm::vec<L, T> result;
    for (i32 i = 0; i < L; i++)
        result[i] = random<T>(min[i], max[i]);

    return result;
}

/*
 * @return A random vector with values in the range in the range [0, 1).
 *
 * @note Uses RANDOM_GENERATOR internally.
 */
template <int L, typename T = f32>
inline glm::vec<L, T> randomVec() {
    return randomVec<L, T>(glm::vec<L, T>(0), glm::vec<L, T>(1));
}

/*
 * @return A random unit vec3.
 *
 * @note Uses RANDOM_GENERATOR internally.
 */
template <int L, typename T = f32>
inline glm::vec<L, T> randomUnitVec() {
    // for vec3 - faster than sampling a gaussian distribution and normalizing - tested with box-muller transform

    while (true) {
        glm::vec<L, T> v = randomVec<L>(glm::vec<L, T>(-1), glm::vec<L, T>(1));
        if (glm::length2(v) <= 1.0)
            return glm::normalize(v);
    }
}

/*
 * @return A random unit vec3 on a unit hemisphere given by a normal.
 *
 * @note Uses RANDOM_GENERATOR internally.
 */
inline vec3 randomVec3OnHemisphere(const vec3& normal) {
    auto v = randomUnitVec<3>();
    return glm::dot(v, normal) >= 0.0f ? v : -v;
}

/*
 * @return A random vec2 in a unit disk around 0.
 *
 * @note Uses RANDOM_GENERATOR internally.
 */
inline vec2 randomVec2InUnitDisk() {
    while (true) {
        auto v = randomVec<2>(vec2(-1), vec2(1));
        if (glm::length2(v) <= 1.0)
            return v;
    }
}
