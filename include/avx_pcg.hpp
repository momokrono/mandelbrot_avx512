#ifndef AVXPCG_HPP
#define AVXPCG_HPP

#include <immintrin.h>
#include <cstdint>
#include <numeric>
#include <array>


class pcg32 {
private:
    std::uint64_t _state{};
    std::uint64_t _stream{1};
public:
    pcg32() = default;
    explicit pcg32(std::uint64_t s, std::uint64_t i = 1) : _state{s}, _stream{i * 2 + 1} { }
    inline constexpr auto seed(std::uint64_t initial_state, std::uint64_t stream = 1) noexcept -> void {
        _state = initial_state;
        _stream = stream * 2 + 1;
        next();
    }
    inline constexpr auto next() noexcept -> std::uint32_t {
        const auto old_state = _state;
        _state = old_state * 6364136223846793005ul + _stream;
        const std::uint32_t xor_shifted = ((old_state >> 18u) ^ old_state) >> 27u;
        const auto rot = old_state >> 59u;
        return (xor_shifted >> rot) | (xor_shifted << ((-rot) & 31));
    }
    inline constexpr auto next(std::uint64_t str) noexcept -> std::uint32_t {
        const std::uint64_t old_state = _state;
        _state = old_state * 6364136223846793005ul + str;
        const std::uint32_t xor_shifted = ((old_state >> 18u) ^ old_state) >> 27u;
        const auto rot = old_state >> 59u;
        return (xor_shifted >> rot) | (xor_shifted << ((-rot) & 31));
    }
    // return float between 0. and 1.
    inline constexpr auto next_f() noexcept -> float {
        auto t = (next() >> 9) | 0x3f800000u;
        return std::bit_cast<float>(t) - 1.0f;
    }
    inline constexpr auto next_f(std::uint64_t str) noexcept -> float {
        auto t = (next(str) >> 9) | 0x3f800000u;
        return std::bit_cast<float>(t) - 1.0f;
    }
    // return double between 0. and 1.
    inline constexpr auto next_d() noexcept -> double {
        auto t = (static_cast<uint64_t>(next()) << 20) | 0x3ff0000000000000ul;
        return std::bit_cast<double>(t) - 1.0f;
    }
    inline constexpr auto next_d(std::uint64_t str) noexcept -> double {
        auto t = (static_cast<uint64_t>(next(str)) << 20) | 0x3ff0000000000000ul;
        return std::bit_cast<double>(t) - 1.0f;
    }
    // TODO jump() and long_jump(), maybe something like xorshiro256?
};


// TODO fix the double.
class avx_pcg32 {
private:
    __m512i _state{};
    __m512i _stream{};

    __attribute__ ((always_inline)) auto advance() noexcept -> __m512i {
        auto const old_state = _state;
        auto const _mul_const = _mm512_set1_epi64(6364136223846793005ul);
        _state = old_state * _mul_const + _stream;
        auto xor_shifted = _mm512_srli_epi64(old_state, 18u);
        xor_shifted = _mm512_xor_epi64(xor_shifted, old_state);
        xor_shifted = _mm512_srli_epi64(xor_shifted, 27u);
        auto const rot = _mm512_srli_epi64(old_state, 59u);
        auto const first = _mm512_srlv_epi64(xor_shifted, rot);
        auto const second = _mm512_rorv_epi64(xor_shifted, -rot);
        return _mm512_or_epi64(first, second);
    }

public:
    avx_pcg32() = default;

    avx_pcg32(__m512i s, __m512i i) : _state{s}, _stream{i * 2 + 1} {}

    inline constexpr auto seed(__m512i initial_state, __m512i stream) noexcept -> void {
        _state = initial_state;
        _stream = stream * 2 + 1;
        advance();
    }

    __attribute__ ((always_inline)) auto next64() noexcept -> __m512i {
        return advance();
    }

    __attribute__ ((always_inline)) auto next32() noexcept -> __m256i {
        return _mm512_castsi512_si256(advance());
    }

    __attribute__ ((always_inline)) auto next_d() noexcept -> __m512d {
        auto t = _mm512_srli_epi64(advance(), 20);
        t = _mm512_or_epi64(t, _mm512_set1_epi64(0x3ff0000000000000ul));
        return _mm512_sub_pd(_mm512_castsi512_pd(t), _mm512_set1_pd(1.));
    }

    __attribute__ ((always_inline)) auto next_f() noexcept -> __m256 {
        return _mm512_cvtpd_ps(next_d());
    }
};

#endif