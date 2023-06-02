#include <SFML/Graphics.hpp>
#include <immintrin.h>
#include <random>

#include "fmt/core.h"
#include "fmt/chrono.h"
#include "task_system.hpp"
#include "avx_mathfun.hpp"


int main()
{
    constexpr auto image_size = 4000;
    constexpr auto screen_size = 1000;
    auto anti_aliasing = 1;
    constexpr auto scale = static_cast<double>( screen_size ) / image_size;
    auto window = sf::RenderWindow( sf::VideoMode( screen_size, screen_size ), "AVX512 Mandel" );

    auto image = sf::Image();
    image.create( image_size, image_size, sf::Color::Black );
    auto texture = sf::Texture();
    auto sprite = sf::Sprite();

    auto min_re = -2.0;
    auto max_re = 1.0;
    auto min_im = -1.5;
    auto max_im = 1.5;
    auto zoom = 1.0f;
    auto max_iter = 256;

    auto mandel = [ & ] ( int start, int end ) -> void {
        auto rg = std::mt19937{std::random_device{}()};
        auto rng = std::uniform_real_distribution<double>(-0.5, 0.5);

        const auto _two = _mm512_set1_pd(2);
        const auto _max_iter = _mm512_set1_epi64(max_iter);
        const auto _brdc = _mm512_setzero_si512();
        const auto _four = _mm512_set1_pd(4);

        for ( auto y{start}; y < end; ++y ) {
            const auto _r_scale = _mm512_set1_pd( ( max_re - min_re ) / image_size );
            const auto _i_scale = _mm512_set1_pd( ( max_im - min_im ) / image_size );
            for ( auto x{0}; x < image_size; x += 8 ) {
                auto red = _mm256_set1_ps(0);
                auto green = _mm256_set1_ps(0);
                auto blue = _mm256_set1_ps(0);
                for ( auto aa{0}; aa < anti_aliasing; ++aa ) {
                    auto _i_0 = _mm512_set1_pd( min_im );
                    auto _r_0 = _mm512_set1_pd( min_re );
                    auto _r_offset = _mm512_set_pd(7., 6., 5., 4., 3., 2., 1., 0.);
                    auto _i_offset = _mm512_set1_pd( y );
                    _r_offset += _mm512_set1_pd( x );
                    auto _rng_offset = _mm512_set_pd(rng(rg), rng(rg), rng(rg), rng(rg),
                                                     rng(rg), rng(rg), rng(rg), rng(rg));
                    _r_offset += _rng_offset;
                    _i_offset += _rng_offset;
                    _r_0 = _mm512_fmadd_pd(_r_scale, _r_offset, _r_0);
                    _i_0 = _mm512_fmadd_pd(_i_scale, _i_offset, _i_0);
                    auto _r_start = _r_0;
                    _r_start = _r_start + _rng_offset * _r_scale;
                    auto _r = _mm512_setzero_pd();
                    auto _i = _mm512_setzero_pd();
                    auto _iter = _mm512_setzero_si512();
                    auto _iter_mask = 0b0;
                    auto _check = 0b0;
                    auto _mod = _mm512_setzero_pd();

                    do {
                        auto _r2 = (_r * _r);
                        auto _i2 = (_i * _i);
                        auto _tr = (_r2 - _i2);
                        _tr = (_tr + _r_start);
                        _i = (_two * _i);
                        _i = _mm512_fmadd_pd( _r, _i, _i_0 );
                        _r = _tr;
                        _mod = (_r2 + _i2);
                        auto _mod_mask = _mm512_cmp_pd_mask( _mod, _four, _CMP_LT_OQ );
                        _iter_mask = _mm512_cmplt_epi64_mask( _iter, _max_iter );
                        _check = _iter_mask & _mod_mask;
                        auto _c = _mm512_mask_set1_epi64( _brdc, _check, 1 );
                        _iter = _iter + _c;
                    } while ( _check > 0 );

                    auto _n = _mm256_set1_ps(0.1) * _mm512_cvtepi64_ps(_iter);
                    const auto _half = _mm256_set1_ps(0.5);
                    auto _red = sin256_ps(_n) * _half;
                    auto _green = sin256_ps(_n + _mm256_set1_ps(2.094)) * _half;
                    auto _blue = sin256_ps(_n + _mm256_set1_ps(4.188)) * _half;
                    _red = (_red + _half);
                    _green = (_green + _half);
                    _blue = (_blue + _half);
                    const auto _255 = _mm256_set1_ps(255);
                    _red = (_red * _255);
                    _green = (_green * _255);
                    _blue = (_blue * _255);
                    red = (red + _red);
                    green = (green + _green);
                    blue = (blue + _blue);
                }
                auto aa = _mm256_set1_ps(static_cast<float>(anti_aliasing));
                red = _mm256_div_ps(red, aa);
                green = _mm256_div_ps(green, aa);
                blue = _mm256_div_ps(blue, aa);

                for ( auto t{0}; t < 8; ++t ) {
                    image.setPixel( x + t, y, sf::Color{ static_cast<uint8_t>(red[t]),
                                                         static_cast<uint8_t>(green[t]),
                                                         static_cast<uint8_t>(blue[t]) } );
                }
            }
        }
    };

    auto needs_update = true;
    auto tasks = task_system();

    auto compute = [ & ] () {
        if ( !needs_update ) { return; }
        fmt::print("max iters: {}\n", max_iter);
        fmt::print("depth: {}\n", zoom);
        fmt::print("AA: {}\n", anti_aliasing);
        for ( auto l{0}; l < image_size; ++l) {
            tasks.async(mandel, l, l+1);
        }
        needs_update = false;
    };

    while ( window.isOpen() ) {
        auto event = sf::Event{};
        while ( window.pollEvent(event) ) {
            switch (event.type) {
                case sf::Event::Closed: {
                    window.close();
                    break;
                }
                case sf::Event::KeyPressed: {
                    if (event.key.code == sf::Keyboard::Escape) {
                        window.close();
                    } else if (event.key.code == sf::Keyboard::P) {
                        anti_aliasing *= 2;
                        needs_update = true;
                    } else if (event.key.code == sf::Keyboard::O) {
                        anti_aliasing = anti_aliasing > 1 ? anti_aliasing / 2 : 1;
                        needs_update = true;
                    } else if (event.key.code == sf::Keyboard::S) {
                        auto r_c = (max_re - min_re) / 2;
                        auto i_c = (max_im - min_im) / 2;
                        image.saveToFile(fmt::format("{}_{}_{}.png", r_c, i_c, max_iter));
                        fmt::print("image saved\n");
                    } else {
                        double w = ( max_re - min_re ) * 0.1;
                        double h = ( max_im - min_im ) * 0.1;

                        if (event.key.code == sf::Keyboard::Left) {
                            min_re -= w, max_re -= w;
                            needs_update = true;
                        }
                        if (event.key.code == sf::Keyboard::Right) {
                            min_re += w, max_re += w;
                            needs_update = true;
                        }
                        if (event.key.code == sf::Keyboard::Up) {
                            min_im -= h, max_im -= h;
                            needs_update = true;
                        }
                        if (event.key.code == sf::Keyboard::Down) {
                            min_im += h, max_im += h;
                            needs_update = true;
                        }
                    }
                    break;
                }
                case sf::Event::MouseButtonPressed: {
                    auto zoomX = [&max_re, &min_re, &max_im, &min_im, &event] (double z) {
                        double x = min_re + (max_re - min_re)*event.mouseButton.x / screen_size;
                        double y = min_im + (max_im - min_im)*event.mouseButton.y / screen_size;
                        double tmp_x = x - (max_re - min_re) / 2 / z;
                        max_re = x + (max_re - min_re) / 2 / z;
                        min_re = tmp_x;
                        double tmp_y = y - (max_im - min_im) / 2 / z;
                        max_im = y + (max_im - min_im) / 2 / z;
                        min_im = tmp_y;
                    };
                    if (event.mouseButton.button == sf::Mouse::Left) {
                        zoomX(2.);
                        zoom *= 2.;
                    }
                    if (event.mouseButton.button == sf::Mouse::Right) {
                        zoomX(0.5);
                        zoom /= 2.;
                        if ( zoom < 0.5 ) { zoom = 0.5; }
                    }
                    needs_update = true;
                    break;
                }
                case sf::Event::MouseWheelScrolled: {
                    if (event.mouseWheelScroll.wheel == sf::Mouse::VerticalWheel) {
                        if (event.mouseWheelScroll.delta > 0) { max_iter *= 2; }
                        else { max_iter /= 2; }
                        if (max_iter < 1) { max_iter = 1; }
                    }
                    needs_update = true;
                    break;
                }
                default:
                    break;
            }
        }

        compute();
        window.clear();
        texture.loadFromImage( image );
        sprite.setTexture( texture );
        sprite.setScale( scale, scale );
        window.draw( sprite );
        window.display();
    }
return 0;
}
