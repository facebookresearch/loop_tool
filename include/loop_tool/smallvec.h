/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <type_traits>

template <class Tp, std::size_t Nm>
class smallvec {
 private:
  static_assert(Nm > 0, "Smallvec only supports non-zero sizes");

 public:
  using value_type = Tp;
  using pointer = value_type *;
  using const_pointer = value_type const *;
  using reference = value_type &;
  using const_reference = value_type const &;
  using iterator = value_type *;
  using const_iterator = value_type const *;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

 private:
  size_type size_ = 0;
  Tp elements_[Nm];

 public:
  constexpr pointer data() noexcept { return const_cast<pointer>(elements_); }
  constexpr const_pointer data() const noexcept {
    return const_cast<const_pointer>(elements_);
  }
  constexpr const_pointer cdata() const noexcept {
    return const_cast<const_pointer>(elements_);
  }

  // clang-format off

    // Iterators
    constexpr iterator       begin()        noexcept { return iterator(data()); }
    constexpr const_iterator begin()  const noexcept { return const_iterator(data()); }
    constexpr const_iterator cbegin() const noexcept { return const_iterator(data()); }
    constexpr iterator       end()          noexcept { return iterator(data() + size()); }
    constexpr const_iterator end()    const noexcept { return const_iterator(data() + size()); }
    constexpr const_iterator cend()   const noexcept { return const_iterator(data() + size()); }

    constexpr reverse_iterator       rbegin()        noexcept { return reverse_iterator(end()); }
    constexpr const_reverse_iterator rbegin()  const noexcept { return const_reverse_iterator(end()); }
    constexpr const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }
    constexpr reverse_iterator       rend()          noexcept { return reverse_iterator(begin()); }
    constexpr const_reverse_iterator rend()    const noexcept { return const_reverse_iterator(begin()); }
    constexpr const_reverse_iterator crend()   const noexcept { return const_reverse_iterator(begin()); }

    // Capacity
    constexpr size_type size() const noexcept { return size_; }
    constexpr size_type max_size() const noexcept { return Nm; }
    [[nodiscard]]
    constexpr bool      empty() const noexcept { return size() == 0; }

    // Element access
    constexpr reference       operator[](size_type n)       noexcept { return elements_[n]; }
    constexpr const_reference operator[](size_type n) const noexcept { return elements_[n]; }

    constexpr reference       front()       noexcept { return elements_[0]; }
    constexpr const_reference front() const noexcept { return elements_[0]; }

    constexpr reference       back()       noexcept { return elements_[size_ - 1]; }
    constexpr const_reference back() const noexcept { return elements_[size_ - 1]; }

  // clang-format on

  constexpr reference at(size_type n) {
    if (n >= size_) {
      throw std::out_of_range(
          "vec::at out of range");  // TODO(zi) provide size() and n;
    }
    return elements_[n];
  }

  constexpr const_reference at(size_type n) const {
    if (n >= size_) {
      throw std::out_of_range(
          "vec::at out of range");  // TODO(zi) provide size() and n;
    }
    return elements_[n];
  }

  constexpr void fill(value_type const &v) { std::fill_n(begin(), size(), v); }

  constexpr void swap(smallvec &other) {
    std::swap_ranges(begin(), end(), other.begin());
    std::swap(size_, other.size_);
  }

 private:
  void destruct_elements() {
    for (size_type i = 0; i < size_; ++i) {
      (elements_ + i)->~Tp();
    }
  }

 public:
  // Constructors
  constexpr smallvec() {}
  ~smallvec() { /*destruct_elements();*/
  }

  constexpr smallvec(smallvec const &other) noexcept(
      std::is_nothrow_copy_constructible<Tp>::value)
      : size_(other.size_) {
    for (size_type i = 0; i < size_; ++i) {
      new (elements_ + i) Tp(other.elements_[i]);
    }
  }

  constexpr smallvec &operator=(smallvec const &other) noexcept(
      std::is_nothrow_copy_constructible<Tp>::value) {
    destruct_elements();
    size_ = other.size_;
    for (size_type i = 0; i < size_; ++i) {
      new (elements_ + i) Tp(other.elements_[i]);
    }
    return *this;
  }

  constexpr smallvec(smallvec &&other) noexcept(
      std::is_nothrow_move_constructible<Tp>::value) {
    destruct_elements();
    size_ = std::exchange(other.size_, 0);
    for (size_type i = 0; i < size_; ++i) {
      new (elements_ + i) Tp(std::move(other.elements_[i]));
    }
  }

  // todo(zi) call operator= on elements when present in both this and other
  constexpr smallvec &operator=(smallvec &&other) noexcept(
      std::is_nothrow_move_constructible<Tp>::value) {
    destruct_elements();
    size_ = other.size_;
    for (size_type i = 0; i < size_; ++i) {
      new (elements_ + i) Tp(std::move(other.elements_[i]));
    }
    return *this;
  }

  constexpr void clear() noexcept {
    destruct_elements();
    size_ = 0;
  }

  constexpr void push_back(Tp const &value) {
    if (size_ >= max_size()) {
      throw std::out_of_range("...");  // TODO(zi) provide size() and n;
    }

    new (elements_ + size_++) Tp(value);
  }

  constexpr void push_back(Tp &&value) {
    if (size_ >= max_size()) {
      throw std::out_of_range("...");  // TODO(zi) provide size() and n;
    }

    new (elements_ + size_++) Tp(std::move(value));
  }

  template <class... Args>
  constexpr reference emplace_back(Args &&...args) {
    if (size_ >= max_size()) {
      throw std::out_of_range("...");  // TODO(zi) provide size() and n;
    }
    new (elements_ + size_++) Tp(std::forward<Args>(args)...);

    return this->operator[](size_ - 1);
  }

  constexpr void pop_back() {
    --size_;
    (elements_ + size_)->~Tp();
  }

  constexpr void resize(size_type count) {
    if (count > max_size()) {
      throw std::out_of_range("...");  // TODO(zi) provide size() and n;
    }

    if (size_ > count) {
      for (size_type i = count; i < size_; ++i) {
        (elements_ + i)->~Tp();
      }
    } else if (size_ < count) {
      for (size_type i = size_; i < count; ++i) {
        new (elements_ + i) Tp;
      }
    }
    size_ = count;
  }

  constexpr void resize(size_type count, value_type const &other) {
    if (count > max_size()) {
      throw std::out_of_range("...");  // TODO(zi) provide size() and n;
    }

    if (size_ > count) {
      for (size_type i = count; i < size_; ++i) {
        (elements_ + i)->~Tp();
      }
    } else if (size_ < count) {
      for (size_type i = size_; i < count; ++i) {
        new (elements_ + i) Tp(other);
      }
    }
    size_ = count;
  }
};
