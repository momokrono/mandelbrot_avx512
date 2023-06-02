// mostly from https://www.youtube.com/watch?v=zULU6Hhp42w, added option to set task parameters
#ifndef TASK_SYSTEM_HPP
#define TASK_SYSTEM_HPP

#include <deque>
#include <thread>
#include <condition_variable>
#include <functional>


using lock_t = std::unique_lock<std::mutex>;

class notification_queue {
    std::deque<std::function<void()>> _q;
    bool _done{false};
    std::mutex _mutex;
    std::condition_variable _ready;

public:
    auto try_pop(std::function<void()>& x) -> bool {
        lock_t lock{_mutex, std::try_to_lock};
        if (!lock || _q.empty()) { return false; }
        x = std::move(_q.front());
        _q.pop_front();
        return true;
    }

    template<typename F>
    auto try_push(F && f) -> bool {
        {
            lock_t lock{_mutex, std::try_to_lock};
            if (!lock) { return false; }
            _q.emplace_back(std::forward<F>(f));
        }
        _ready.notify_one();
        return true;
    }

    auto done() -> void {
        {
            lock_t lock{_mutex};
            _done = true;
        }
        _ready.notify_all();
    }

    auto pop(std::function<void()>& x) -> bool {
        lock_t lock{_mutex};
        while ( _q.empty() && !_done ) _ready.wait(lock);
        if ( _q.empty() ) return false;
        x = std::move( _q.front() );
        _q.pop_front();
        return true;
    }

    template<typename F, typename ...Args>
    auto push(F && f, Args... args) -> void {
        {
            lock_t lock{_mutex};
            _q.emplace_back( [fn = std::forward<F>(f), args = std::tuple{std::forward<Args>(args)...} ] {
                return std::apply(std::move(fn), args);
            });
        }
        _ready.notify_one();
    }
};


class task_system {
    const unsigned _count{std::thread::hardware_concurrency()};
    std::vector<std::thread> _threads;
    std::vector<notification_queue> _q{_count};
    std::atomic<unsigned> _index{0};

    auto run(unsigned i) -> void {
        while ( true ) {
            auto f = std::function<void()>{};
            for ( unsigned n = 0; n != _count * 32; ++n ) {
                if ( _q[ (i + n) % _count] .try_pop(f) ) { break; }
            }
            if ( !f && !_q[i].pop(f) ) { break; }

            f();
        }
    }

public:
    task_system() {
        for ( unsigned n = 0; n != _count; ++n ) {
            _threads.emplace_back( [&, n] { run(n); } );
        }
    }

    ~task_system() {
        for ( auto& e : _q ) e.done();
        for ( auto& e : _threads ) e.join();
    }

    template<typename F, typename ...Args>
    auto async(F && f, Args... args) -> void {
        auto i = _index++;

       for ( unsigned n = 0; n != _count; ++n ) {
            if ( _q[ (i + n) % _count ].try_push(
                    [ fn = std::forward<F>(f), args = std::tuple{std::forward<Args>(args)...} ] {
                        return std::apply(std::move(fn), args);
                    } ) ) { return; }
        }

        _q[ i % _count ].push(
                [ fn = std::forward<F>(f), args = std::tuple{std::forward<Args>(args)...} ] {
                    return std::apply(std::move(fn), args); } );
    }
};


#endif
