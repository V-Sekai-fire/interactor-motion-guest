// Kimodo's motion decode (vendor/kimodo-ggml/src/motion_decode.cpp) kept in
// matrices: the same arithmetic in the same order up to the global 6D ->
// matrix -> parent-local matrix step, where upstream then converts to a
// quaternion. AGENTS.md rule 11 keeps rotations as matrices, so the guest and
// the native gate tool both decode with this header and never see a
// quaternion. Header-only, plain C++.
#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace motion::kimodo_decode {

struct M {
	float v[9];
};

inline M mul(const M &a, const M &b) {
	M r{};
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			for (int k = 0; k < 3; ++k)
				r.v[i * 3 + j] += a.v[i * 3 + k] * b.v[k * 3 + j];
	return r;
}

inline M tr(const M &a) {
	M r{};
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			r.v[i * 3 + j] = a.v[j * 3 + i];
	return r;
}

// Upstream's six(): columns a, b, z of a 6D pair, Gram-Schmidt.
inline M six(const float *x) {
	float n = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
	float a[3] = { x[0] / n, x[1] / n, x[2] / n };
	float z[3] = { a[1] * x[5] - a[2] * x[4], a[2] * x[3] - a[0] * x[5], a[0] * x[4] - a[1] * x[3] };
	n = std::sqrt(z[0] * z[0] + z[1] * z[1] + z[2] * z[2]);
	for (float &v : z)
		v /= n;
	float b[3] = { z[1] * a[2] - z[2] * a[1], z[2] * a[0] - z[0] * a[2], z[0] * a[1] - z[1] * a[0] };
	return M{ { a[0], b[0], z[0], a[1], b[1], z[1], a[2], b[2], z[2] } };
}

// normalized: frames x D (D = 9 + 12 J), the sampler's output. Writes
// root (frames x 3) and local (frames x J x 9, row-major, parent-relative).
inline bool decode(const float *x, size_t T, size_t J, const int *parents, const float *gm, const float *gs,
		const float *bm, const float *bs, std::vector<float> &root, std::vector<float> &local) {
	const size_t D = 9 + 12 * J, body = D - 5, rotation = 5 + 3 * J;
	auto scale = [](float v) { return std::sqrt(v * v + 1.e-5f); };
	std::vector<float> f(D);
	std::vector<M> g(J);
	root.assign(T * 3, 0.f);
	local.assign(T * J * 9, 0.f);
	for (size_t t = 0; t < T; ++t) {
		const float *in = x + t * D;
		for (size_t i = 0; i < 5; ++i)
			f[i] = in[i] * scale(gs[i]) + gm[i];
		for (size_t i = 0; i < body; ++i)
			f[5 + i] = in[5 + i] * scale(bs[i]) + bm[i];
		root[t * 3] = f[0] + f[5];
		root[t * 3 + 1] = f[6];
		root[t * 3 + 2] = f[2] + f[7];
		for (size_t j = 0; j < J; ++j)
			g[j] = six(f.data() + rotation + j * 6);
		for (size_t j = 0; j < J; ++j) {
			const M l = parents[j] < 0 ? g[j] : mul(tr(g[static_cast<size_t>(parents[j])]), g[j]);
			for (int k = 0; k < 9; ++k)
				local[(t * J + j) * 9 + k] = l.v[k];
		}
	}
	return true;
}

} // namespace motion::kimodo_decode
