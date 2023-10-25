#include <SFML/Graphics.hpp>
#include <immintrin.h>
#include <algorithm>
#include <random>
#include <latch>

#include "fmt/core.h"
#include "fmt/chrono.h"
#include "avx_mathfun.hpp"
#include "avx_pcg.hpp"
#include "spl/image.hpp"
#include "task_system.hpp"


int main()
{
    auto test=[]() -> std::string_view {
        std::string stringa = "hello";
        return std::string_view{stringa};
    };
    auto boh = test();
    fmt::print("{}\n", boh);
    constexpr auto image_size = 1000;
    auto render_factor = 4;
    auto high_res_render = false;
    auto colored_pic = true;
    auto first_color = true;
    auto aborted = false;
    auto anti_aliasing = 1;
    auto window = sf::RenderWindow( sf::VideoMode( image_size, image_size ), "AVX512 Mandel" );

    // the sf::Image unfortunately does not allow direct write access to its vector of pixels.
    // so what I will do is to use this image as the one the gui will display, and instead use a spl::graphics::image
    // as buffer for rendering, since I can use std::copy to dump whatever the compute threads render
    // directly into its rbg vector.
    // then, once the buffer is ready, I will re-create the image directly from the buffer.
    auto image = sf::Image();
    // these two down here are used to render the image onto the screen:
    // the texture will load the image, and the sprite created from the texture will be displayed by the window.
    auto texture = sf::Texture();
    auto sprite = sf::Sprite();

    auto min_re = -2.0;
    auto max_re = 1.0;
    auto min_im = -1.5;
    auto max_im = 1.5;
    auto zoom = 1.0f;
    auto max_iter = 256;

    auto tasks = task_system();
    // this semaphore will be used by the gui thread to signal the compute thread to render the new frame since
    // there is no need to compute the render each and every frame unless something changed, like zoom, AA, etc...
    auto needs_update = std::binary_semaphore{1};
    auto done_rendering = false;
    auto line_count = std::atomic<int>{};

    // this is the lambda that will compute, row by row, the fractal
    auto mandel_avx512 = [ & ] ( spl::graphics::image & buffer, int line ) -> void {
        auto temp_buffer = std::vector<spl::graphics::rgba>{};
        temp_buffer.reserve(buffer.width());
        thread_local auto rng = pcg32{};

        const auto _two = _mm512_set1_pd(2);
        const auto _max_iter = _mm512_set1_epi64(max_iter);
        const auto _brdc = _mm512_setzero_si512();
        const auto _escape_radius = _mm512_set1_pd(1000);
        const auto _255 = _mm256_set1_ps(255);

        const auto _r_scale = _mm512_set1_pd((max_re - min_re) / static_cast<double>(buffer.width()));
        const auto _i_scale = _mm512_set1_pd((max_im - min_im) / static_cast<double>(buffer.width()));
        // I'm subtracting 0.5 since the double generated is between 0 and 1, and I want +/- half the pixel size
        auto _x_rng_offset = std::vector<__m512d>{};
        auto _y_rng_offset = std::vector<__m512d>{};
        _x_rng_offset.reserve(anti_aliasing);
        _y_rng_offset.reserve(anti_aliasing);
        for ( auto aa{0}; aa < anti_aliasing; ++aa ) {
            _x_rng_offset.push_back(_mm512_set1_pd(rng.next_d()-0.5));
            _y_rng_offset.push_back(_mm512_set1_pd(rng.next_d()-0.5));
        }
        // we move horizontally by 8 since we are computing 8 doubles at a time
        for ( auto x{0u}; x < buffer.width(); x += 8 ) {
            auto red = _mm256_set1_ps(0);
            auto green = _mm256_set1_ps(0);
            auto blue = _mm256_set1_ps(0);
            // the way I compute AA on this fractal is by doing something similar to what it's done with ray-tracing:
            // basically I compute the color of a certain number of complex numbers around the one at the center of the
            // pixel, and I average it after the for loop.
            for ( auto aa{0}; aa < anti_aliasing; ++aa ) {
                auto _i_0 = _mm512_set1_pd( min_im );
                auto _r_0 = _mm512_set1_pd( min_re );
                auto _r_offset = _mm512_set_pd(7., 6., 5., 4., 3., 2., 1., 0.);
                auto _i_offset = _mm512_set1_pd( line );
                _r_offset += _mm512_set1_pd( x );
                _r_offset += _x_rng_offset[aa];
                _i_offset += _y_rng_offset[aa];
                _r_0 = _mm512_fmadd_pd(_r_scale, _r_offset, _r_0);
                _i_0 = _mm512_fmadd_pd(_i_scale, _i_offset, _i_0);
                auto _r_start = _r_0;
                _r_start = _r_start + _x_rng_offset[aa] * _r_scale;
                auto _r = _mm512_setzero_pd();
                auto _i = _mm512_setzero_pd();
                auto _iter = _mm512_setzero_si512();
                auto _iter_mask = 0b0;
                auto _mod_mask = 0b0;
                auto _check = 0b0;
                auto _mod = _mm512_setzero_pd();
                // the idea inside this loop is:
                // we store all the x and y values of the 8 complex numbers and apply the usual mandelbrot steps.
                // we store all the iterations and abs of out points.
                // we compare after one iteration if any point escapes generating a mask set for each point not escaped.
                // we also check if the current iteration is greater than the max, and we generate a mask set for each
                // point whose iteration is less than the max.
                // basically, we are saying "a point is still valid if it's inside both the escape time and radius" so
                // we bit-wise _and_ the two masks to check if any of the two loop condition are *not* verified.
                // when a point fails at least one of the two condition, the relative mask bit will be set to 0 and
                // since the mask is a simple 8-bit unsigned number, if the mask is 0 it means _all_ 8 points failed
                // at least one of the two condition, and we exit the loop.
                // if we are still looping, meaning at least 1 point is valid, we update the iteration counter
                // and the new absolute value only for the valid ones.
                do {
                    auto _r2 = (_r * _r);
                    auto _i2 = (_i * _i);
                    auto _tr = (_r2 - _i2);
                    _tr = (_tr + _r_start);
                    _i = (_two * _i);
                    _i = _mm512_fmadd_pd( _r, _i, _i_0 );
                    _r = _tr;
                    auto _tmp_mod = (_r2 + _i2);
                    _mod_mask = _mm512_cmp_pd_mask( _tmp_mod, _escape_radius, _CMP_LT_OQ );
                    _iter_mask = _mm512_cmplt_epi64_mask( _iter, _max_iter );
                    _check = _iter_mask & _mod_mask;
                    auto _c = _mm512_mask_set1_epi64( _brdc, _check, 1 );
                    _iter = _iter + _c;
                    _mod = _mm512_mask_blend_pd(_mod_mask, _mod, _tmp_mod);
                } while ( _check > 0 );

                // two coloring algorithms found online, feel free to change them!
                // the first one is picked from the javidx9 YouTube video that inspired the project
                // and despite beautiful colors it's affected by banding and noise, meaning two adjacent points could
                // have completely different colors, making the pictures quite ugly near singular points.
                if ( colored_pic ) {
                    if ( first_color ) {
                        auto _n = _mm256_set1_ps(0.1) * _mm512_cvtepi64_ps(_iter);
                        const auto _half = _mm256_set1_ps(0.5);
                        auto _red = sin256_ps(_n) * _half;
                        auto _green = sin256_ps(_n + _mm256_set1_ps(2.094)) * _half;
                        auto _blue = sin256_ps(_n + _mm256_set1_ps(4.188)) * _half;
                        _red = (_red + _half);
                        _green = (_green + _half);
                        _blue = (_blue + _half);
                        _red = (_red * _255);
                        _green = (_green * _255);
                        _blue = (_blue * _255);
                        red = (red + _red);
                        green = (green + _green);
                        blue = (blue + _blue);
                    } else {
                        // if all points reached max iter we can skip all the computation for the colors
                        // and go straight to the next AA pass
                        if (_iter_mask == 0b0) {
                            red += _mm256_set1_ps(64);
                            green += _mm256_set1_ps(64);
                            blue += _mm256_set1_ps(64);
                            continue;
                        }
                        __m512i tmp_red;
                        __m512i tmp_green;
                        __m512i tmp_blue;
                        _iter += _mm512_set1_epi64(2);
                        const auto _log2 = _mm256_set1_ps(std::log(2.f));
                        auto _log = log256_ps(_mm512_cvtpd_ps(_mod));
                        _log = log256_ps(_log);
                        auto _final_iters = _mm512_cvtepi64_ps(_iter) - _log / _log2;
                        _final_iters = _mm256_max_ps(_final_iters, _mm256_set1_ps(0));
                        auto periodic_color = [&](int c) {
                            if (c < 128) return 128 + c;
                            else if (c < 384) return 383 - c;
                            return c - 384;
                        };
                        for (auto t{0}; t < 8; ++t) {
                            auto a = std::sqrt(_final_iters[t]) * 8;
                            tmp_red[t] = periodic_color(static_cast<int>(floor(a * 2)) % 512);
                            tmp_green[t] = periodic_color(static_cast<int>(floor(a * 3)) % 512);
                            tmp_blue[t] = periodic_color(static_cast<int>(floor(a * 5)) % 512);
                        }
                        tmp_red = _mm512_mask_blend_epi64(_iter_mask, _mm512_set1_epi64(64), tmp_red);
                        tmp_green = _mm512_mask_blend_epi64(_iter_mask, _mm512_set1_epi64(64), tmp_green);
                        tmp_blue = _mm512_mask_blend_epi64(_iter_mask, _mm512_set1_epi64(64), tmp_blue);
                        red += _mm512_cvtepi64_ps(tmp_red);
                        green += _mm512_cvtepi64_ps(tmp_green);
                        blue += _mm512_cvtepi64_ps(tmp_blue);
                    }
                }
                // this other algorithm is the classic mandelbrot black and white, it has excellent smooth blending but
                // with the way I handle iterations (basically << 1 or >> 1) I don't have much control over the shadow
                // and the overall image it's either too bright or too dark, and thus details are not so visible.
                // also, I'm using the dumb way to make BW pixels, basically (r,r,r), and the human eye doesn't perceive
                // each r-g-b color with the same sensitivity, so I should change the way the final rgb pixel is made.
                else {
                    _iter += _mm512_set1_epi64(1);
                    const auto _log2 = _mm256_set1_ps(std::log(2.f));
                    auto _log = log256_ps(_mm512_cvtpd_ps(_mod));
                    _log = log256_ps(_log);
                    auto _final_iters = _mm512_cvtepi64_ps(_iter) - _log / _log2;
                    auto frac = _final_iters / _mm512_cvtepi64_ps(_max_iter);
                    auto stability = _mm256_min_ps(frac, _mm256_set1_ps(1.0));
                    stability = _mm256_max_ps(stability, _mm256_setzero_ps());
                    red += (_mm256_set1_ps(1) - stability) * _255;
                    green = red;
                    blue = red;
                }
            }
            auto aa = _mm256_set1_ps(static_cast<float>(anti_aliasing));
            red = _mm256_div_ps(red, aa);
            green = _mm256_div_ps(green, aa);
            blue = _mm256_div_ps(blue, aa);
            // you can think of every _mmXXX as a simple array of N, so you can just use the [] operator
            for ( auto t{0}; t < 8; ++t ) {
                temp_buffer.emplace_back(static_cast<uint8_t>(red[t]),
                                         static_cast<uint8_t>(green[t]),
                                         static_cast<uint8_t>(blue[t]));
            }
        }
        std::copy(temp_buffer.begin(), temp_buffer.end(), buffer.get_pixel_iterator(0, line));
        ++line_count;
    };

    // I don't like the way this lambda is organized, at all.
    auto compute = [ & ] ( std::stop_token const & stop ) {
        while ( !stop.stop_requested() ) {
            needs_update.acquire();
            // since we wait for an acquire of the update flag, once we destroy the jthread and request a stop, the
            // thread will be still waiting on this flag.
            // so, in order to exit the loop, we first request the stop and then unblock the flag and then
            // check for the stop in order to exit the program.
            if ( stop.stop_requested() ) { break; }
            if ( aborted ) {
                tasks.clear();
                aborted = false;
                fmt::print("aborted\n");
                continue;
            }
            auto render_dim = image_size;
            if ( high_res_render ) {
                render_dim *= render_factor;
                anti_aliasing *= render_factor;
            }
            fmt::print("max iters: {}\n", max_iter);
            fmt::print("depth: {}\n", zoom);
            fmt::print("size: {}\n", render_dim);
            fmt::print("AA: {}\n", anti_aliasing);
            line_count = 0;
            auto image_buffer = spl::graphics::image(render_dim, render_dim);
            for (auto line{0u}; line < image_buffer.height(); ++line) {
                tasks.async(mandel_avx512, std::ref(image_buffer), line);
            }
            auto last_line = 0;
            auto start_time = std::chrono::steady_clock::now();
            while ( line_count < render_dim ) {
                auto current_line = line_count.load(std::memory_order_relaxed);
                if ( current_line == last_line ) { continue; }
                auto progress = current_line * 100 / render_dim;
                if ( current_line % 100 == 0 ) { fmt::print("progress: {}\n", progress); }
                last_line = current_line;
            }
            auto end_time = std::chrono::steady_clock::now();
            if ( high_res_render ) {
                high_res_render = false;
                anti_aliasing /= render_factor;
                fmt::print("high res render done in {}\n",
                           std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time));
                auto r_c = (max_re - min_re) / 2;
                auto i_c = (max_im - min_im) / 2;
                auto filename = fmt::format("{}_{}_{}_{}.png",
                                            r_c, i_c, max_iter, colored_pic ? "color" : "bw");
                image_buffer.save_to_file(filename);
                fmt::print("image saved with name {}\n\n", filename);
            } else {
                const auto *first_pxl = &(image_buffer.raw_data()->r);
                image.create(image_size, image_size, first_pxl);
                texture.loadFromImage(image);
                sprite.setTexture(texture);
                fmt::print("render done in {}\n\n",
                           std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time));
            }
            done_rendering = true;
        }
    };

    auto signal_update = [&](){
            needs_update.release();
            done_rendering = false;
    };

    // the main thing to do inside this lambda is setting the update flag for the compute thread when an event
    // that changes the frame occurs.
    auto handle_gui = [ & ] () {
        auto event = sf::Event{};
        while ( window.pollEvent(event) ) {
            if ( event.type == sf::Event::Closed ) {
                window.close();
                break;
            }
            if ( done_rendering ) {
                switch (event.type) {
                    case sf::Event::KeyPressed: {
                        if (event.key.code == sf::Keyboard::Escape) {
                            window.close();
                        } else if (event.key.code == sf::Keyboard::P) {
                            anti_aliasing *= 2;
                            signal_update();
                        } else if (event.key.code == sf::Keyboard::O) {
                            anti_aliasing = anti_aliasing > 1 ? anti_aliasing / 2 : 1;
                            signal_update();
                        } else if (event.key.code == sf::Keyboard::C) {
                            colored_pic = !colored_pic;
                            signal_update();
                        } else if (event.key.code == sf::Keyboard::X) {
                                first_color = !first_color;
                                signal_update();
                        } else if (event.key.code == sf::Keyboard::B) {
                            line_count = image_size * render_factor;
                            aborted = true;
                            fmt::print("aborting computation\n");
                            signal_update();
                        } else if (event.key.code == sf::Keyboard::R) {
                            fmt::print("starting high res render\n");
                            high_res_render = true;
                            signal_update();
                        } else if (event.key.code == sf::Keyboard::S) {
                            auto r_c = (max_re - min_re) / 2;
                            auto i_c = (max_im - min_im) / 2;
                            image.saveToFile(fmt::format("{}_{}_{}_{}.png",
                                                         r_c, i_c, max_iter, colored_pic ? "color" : "bw"));
                            fmt::print("image saved\n\n");
                        } else {
                            double w = (max_re - min_re) * 0.1;
                            double h = (max_im - min_im) * 0.1;

                            if (event.key.code == sf::Keyboard::Left) {
                                min_re -= w, max_re -= w;
                            }
                            if (event.key.code == sf::Keyboard::Right) {
                                min_re += w, max_re += w;
                            }
                            if (event.key.code == sf::Keyboard::Up) {
                                min_im -= h, max_im -= h;
                            }
                            if (event.key.code == sf::Keyboard::Down) {
                                min_im += h, max_im += h;
                            }
                            signal_update();
                        }
                        break;
                    }
                    case sf::Event::MouseButtonPressed: {
                        auto zoomX = [&](double z) {
                            double x = min_re + (max_re - min_re) * event.mouseButton.x / image_size;
                            double y = min_im + (max_im - min_im) * event.mouseButton.y / image_size;
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
                            if (zoom < 0.5) { zoom = 0.5; }
                        }
                        signal_update();
                        break;
                    }
                    case sf::Event::MouseWheelScrolled: {
                        if (event.mouseWheelScroll.wheel == sf::Mouse::VerticalWheel) {
                            if (event.mouseWheelScroll.delta > 0) { max_iter *= 2; }
                            else { max_iter /= 2; }
                            if (max_iter < 1) { max_iter = 1; }
                        }
                        signal_update();
                        break;
                    }
                    default:
                        break;
                }
            }
        }
    };

    auto com = std::jthread{compute};
    fmt::print("Simple mandelbrot plotter, using AVX512 intrinsics.\n"
               "Below are the available controls:\n"
               "- arrow keys : pan the view\n"
               "- left mouse click : zoom in\n"
               "- right mouse click : zoom out\n"
               "- mouse wheel up : increase iterations\n"
               "- mouse wheel down : decrease iterations\n"
               "- s : save the current image\n"
               "- r : render a {0}x image with {0}x AA and save it\n"
               "- o : decrease the anti aliasing level\n"
               "- p : increase the anti aliasing level\n"
               "- c : switch between black and white and colored\n"
               "- x : switch between coloring algorithm\n"
               "- b : to abort the current computation\n"
               "\n", render_factor);

    while ( window.isOpen() ) {
        handle_gui();
        window.clear();
        window.draw(sprite);
        window.display();
    }
    line_count = image_size * render_factor;
    com.request_stop();
    needs_update.release(); // <--- if I don't release this the compute thread will hold onto the acquire and never exit
                            // ask me how I know
    fmt::print("bye!\n");
    return 0;
}
