// Kimodo's initial diffusion noise, the same for a seed on every platform.
//
// Upstream (kimodo::model::generate_embedding) draws it with
// std::mt19937_64 and std::normal_distribution<float>. The engine is fixed by
// the standard; the distribution is not, and the guest's libstdc++ 14 and
// the host's give different noise for one seed (gates/10-motion: noise[0]
// -0.516 in the guest, 0.566 on the host), which AGENTS.md already records
// for std::shuffle and the uniform distributions (Gate 4). So motion.elf and
// tools/motion_native both draw here: the engine's raw words, 53-bit
// uniforms, Marsaglia's polar method in double, pairs in (y, x) order.
// Header-only, plain C++.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

namespace motion {

inline void normal_noise(uint64_t seed, float *out, size_t n) {
	std::mt19937_64 rng(seed);
	auto uniform = [&rng] { return double(rng() >> 11) * 0x1.0p-53; };
	size_t i = 0;
	while (i < n) {
		double x, y, r2;
		do {
			x = 2.0 * uniform() - 1.0;
			y = 2.0 * uniform() - 1.0;
			r2 = x * x + y * y;
		} while (r2 > 1.0 || r2 == 0.0);
		const double m = std::sqrt(-2.0 * std::log(r2) / r2);
		out[i++] = float(y * m);
		if (i < n) {
			out[i++] = float(x * m);
		}
	}
}

} // namespace motion
