#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <span>

extern "C" {
float* static_relu1(float*);
float* static_relu3(float*);
float* dynamic_relu1(uint64_t, float*);
float* dynamic_relu3(uint64_t, float*);
float dynamic_sum1(uint64_t, float*);
float* dynamic_sum3(uint64_t, float*);
float scalar_roundtrip(float);
float* literal_relu();
}

static bool equal(std::span<const float> got, std::span<const float> want) {
    for (size_t i = 0; i != want.size(); ++i) {
        if (got[i] != want[i]) {
            std::cerr << "element " << i << ": expected " << want[i] << ", got " << got[i] << '\n';
            return false;
        }
    }
    return true;
}

int main() {
    std::array<float, 4> one = {-3, 0, 2, 5};
    std::array<float, 4> one_want = {0, 0, 2, 5};
    std::array<float, 6> three = {-3, 0, 2, 5, -1, 4};
    std::array<float, 6> three_want = {0, 0, 2, 5, 0, 4};
    bool ok = equal({static_relu1(one.data()), 4}, one_want);
    ok &= equal({static_relu3(three.data()), 6}, three_want);
    for (uint64_t n : {uint64_t(1), uint64_t(2)}) {
        ok &= equal({dynamic_relu1(n, one.data()), n}, std::span(one_want).first(n));
        ok &= equal({dynamic_relu3(n, three.data()), n * 3}, std::span(three_want).first(n * 3));
        float sum = 0;
        for (size_t i = 0; i != n; ++i) sum += one[i];
        ok &= dynamic_sum1(n, one.data()) == sum;
        std::array<float, 2> sums = {-1, 8};
        ok &= equal({dynamic_sum3(n, three.data()), n}, std::span(sums).first(n));
    }
    ok &= scalar_roundtrip(-7) == -7;
    std::array<float, 3> literal_want = {0, 0, 5};
    ok &= equal({literal_relu(), 3}, literal_want);
    return ok ? 0 : 1;
}
