// A value or an error, for toolchains whose standard library lacks <expected>.
#pragma once

#include <type_traits>
#include <utility>
#include <variant>

namespace kimodo {

template <class E> class unexpected {
public:
    explicit unexpected(E error) : error_(std::move(error)) {}
    E &error() & { return error_; }
    E &&error() && { return std::move(error_); }

private:
    E error_;
};

template <class E> unexpected(E) -> unexpected<E>;

template <class T, class E> class expected {
public:
    expected(T value) : v_(std::in_place_index<0>, std::move(value)) {}
    template <class G> expected(unexpected<G> error) : v_(std::in_place_index<1>, E(std::move(error).error())) {}

    bool has_value() const { return v_.index() == 0; }
    explicit operator bool() const { return has_value(); }
    T &operator*() & { return std::get<0>(v_); }
    const T &operator*() const & { return std::get<0>(v_); }
    T &&operator*() && { return std::get<0>(std::move(v_)); }
    T *operator->() { return &std::get<0>(v_); }
    const T *operator->() const { return &std::get<0>(v_); }
    T &value() & { return std::get<0>(v_); }
    const E &error() const & { return std::get<1>(v_); }
    E &&error() && { return std::get<1>(std::move(v_)); }

private:
    std::variant<T, E> v_;
};

template <class E> class expected<void, E> {
public:
    expected() = default;
    template <class G> expected(unexpected<G> error) : ok_(false), error_(std::move(error).error()) {}

    bool has_value() const { return ok_; }
    explicit operator bool() const { return ok_; }
    const E &error() const & { return error_; }

private:
    bool ok_ = true;
    E error_{};
};

} // namespace kimodo
