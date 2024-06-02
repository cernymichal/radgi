#pragma once

/*
https://en.cppreference.com/w/c/types/integer
https://en.cppreference.com/w/cpp/language/types
*/

// TODO C++23 fixed width floating point types
// #include <stdfloat>

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;
#ifdef __SIZEOF_INT128__  // Only 64-bit GCC and Clang
typedef __int128 i128;
#endif
// typedef intmax_t imax;

// Fastest signed integer types with at least that size
typedef int_fast8_t i8f;
typedef int_fast16_t i16f;
typedef int_fast32_t i32f;
typedef int_fast64_t i64f;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#ifdef __SIZEOF_INT128__  // Only 64-bit GCC and Clang
typedef unsigned __int128 u128;
#endif
// typedef uintmax_t umax; // CUDA collision

// Fastest u32 integer types with at least that size
typedef uint_fast8_t u8f;
typedef uint_fast16_t u16f;
typedef uint_fast32_t u32f;
typedef uint_fast64_t u64f;

typedef float f32;
typedef double f64;
#ifdef __SIZEOF_FLOAT128__  // Only 64-bit GCC and Clang
typedef __float128 f128;
#endif
