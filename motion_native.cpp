// motion_native: Gate 10's flat control and the Kimodo stand-in's embedder,
// on the HOST (AGENTS.md rule 10: the oracle for a graph this size runs on the
// host, under `timeout 300`). It compiles the ports' OWN sources from their
// org checkouts (V-Sekai-fire/kimodo-ggml, V-Sekai-fire/motion-bricks-ggml:
// KIMODO_SRC and MB_SRC in CMakeLists.txt) against the org's ggml
// (vendor/ggml), host ggml-cpu, nothing of motion.elf's.
//
//   motion_native embed   BUNDLE OUTDIR key=prompt [key=prompt ...]
//       kimodo-ggml's LLM2Vec encoder, one OUTDIR/<key>.f32 (4096 floats) each
//   motion_native kimodo  MOTION.gguf EMB.f32 SECONDS SEED STEPS OUTDIR
//       kimodo::model::generate_embedding's path (mt19937_64 noise, the DDIM
//       sampler, CFG 2/2); writes raw.f32 (frames x 369) and transforms.f32
//   motion_native mbricks BUNDLE STYLE.mbstyle SECONDS SEED OUTDIR
//       motion-bricks.cpp's C API in the loop of character-fox's
//       motionbricks_motion.pose_tracks (speed 1.2, turn); writes transforms.f32
//   motion_native compare A.f32 B.f32 [tolerance]
//       n, max |a - b|, its index, RMS; with a tolerance, WITHIN (exit 0) or
//       OUTSIDE (exit 1)
//
// transforms.f32 is motion.elf's motion_result layout: per frame, per joint,
// the parent-local row-major 3x3 then the translation (the root's is the root
// position, the others their rest offset).
#include <kimodo/kimodo.hpp>
#include <motionbricks/motionbricks.h>

#include "denoiser.hpp"
#include "ggml_weights.hpp"
#include "kimodo_decode.h"
#include "llm_text_encoder.hpp"
#include "noise.h"
#include "skeleton.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::vector<float> read_f32(const fs::path &p) {
	std::ifstream in(p, std::ios::binary | std::ios::ate);
	if (!in) throw std::runtime_error("cannot read " + p.string());
	const auto n = in.tellg();
	std::vector<float> v(size_t(n) / sizeof(float));
	in.seekg(0);
	in.read(reinterpret_cast<char *>(v.data()), std::streamsize(v.size() * sizeof(float)));
	return v;
}

void write_f32(const fs::path &p, const std::vector<float> &v) {
	std::ofstream out(p, std::ios::binary | std::ios::trunc);
	out.write(reinterpret_cast<const char *>(v.data()), std::streamsize(v.size() * sizeof(float)));
	if (!out) throw std::runtime_error("cannot write " + p.string());
}

int embed(int argc, char **argv) {
	if (argc < 5) throw std::runtime_error("embed BUNDLE OUTDIR key=prompt ...");
	auto enc = kimodo::detail::llm_text_encoder::load(argv[2]);
	if (!enc) throw std::runtime_error(enc.error());
	fs::create_directories(argv[3]);
	for (int i = 4; i < argc; ++i) {
		const std::string kv = argv[i];
		const size_t eq = kv.find('=');
		if (eq == std::string::npos) throw std::runtime_error("expected key=prompt, got " + kv);
		auto e = (*enc)->encode(kv.substr(eq + 1));
		if (!e) throw std::runtime_error(e.error());
		write_f32(fs::path(argv[3]) / (kv.substr(0, eq) + ".f32"), std::vector<float>(e->begin(), e->end()));
		std::printf("embed %s '%s' e[0..2] %.9g %.9g %.9g\n", kv.substr(0, eq).c_str(), kv.substr(eq + 1).c_str(),
				(*e)[0], (*e)[1], (*e)[2]);
	}
	return 0;
}

int kimodo_run(int argc, char **argv) {
	if (argc != 8) throw std::runtime_error("kimodo MOTION.gguf EMB.f32 SECONDS SEED STEPS OUTDIR");
	const auto emb = read_f32(argv[3]);
	const double seconds = std::stod(argv[4]);
	const uint64_t seed = std::stoull(argv[5]);
	const unsigned steps = unsigned(std::stoul(argv[6]));
	if (emb.size() != 4096) throw std::runtime_error("embedding is not 4096 floats");
	auto w = kimodo::detail::ggml_motion_weights::load(argv[2]);
	if (!w) throw std::runtime_error(w.error());
	const auto *skel = kimodo::detail::find_skeleton((*w)->skeleton_key());
	const size_t D = (*w)->motion_dim(), J = skel->joints();
	const size_t frames = size_t(std::lround(seconds * 30.0));
	// motion.elf's noise for the seed (noise.h: upstream's
	// std::normal_distribution differs between standard libraries).
	std::vector<float> noise(frames * D);
	motion::normal_noise(seed, noise.data(), noise.size());
	std::printf("kimodo %zu frames, %u steps, seed %llu, noise[0..2] %.9g %.9g %.9g\n", frames, steps,
			(unsigned long long)seed, noise[0], noise[1], noise[2]);
	auto sampled = kimodo::detail::sample_motion_from_noise(**w, noise, emb, frames, steps, 2.f, 2.f);
	if (!sampled) throw std::runtime_error(sampled.error());
	auto gm = (*w)->f32_values("stats.global_root.mean"), gs = (*w)->f32_values("stats.global_root.std");
	auto bm = (*w)->f32_values("stats.body.mean"), bs = (*w)->f32_values("stats.body.std");
	if (!gm || !gs || !bm || !bs) throw std::runtime_error("missing stats");
	std::vector<float> root, local;
	motion::kimodo_decode::decode(sampled->data(), frames, J, skel->parents.data(), gm->data(), gs->data(), bm->data(),
			bs->data(), root, local);
	std::vector<float> xf(frames * J * 12);
	for (size_t f = 0; f < frames; ++f)
		for (size_t j = 0; j < J; ++j) {
			float *o = &xf[(f * J + j) * 12];
			std::memcpy(o, &local[(f * J + j) * 9], 9 * sizeof(float));
			for (int k = 0; k < 3; ++k) o[9 + k] = skel->parents[j] < 0 ? root[f * 3 + size_t(k)] : skel->offsets[j][size_t(k)];
		}
	fs::create_directories(argv[7]);
	write_f32(fs::path(argv[7]) / "raw.f32", *sampled);
	write_f32(fs::path(argv[7]) / "transforms.f32", xf);
	std::printf("kimodo wrote %zu raw, %zu transform floats\n", sampled->size(), xf.size());
	return 0;
}

void check(mb_status s, const char *what, const char *eb) {
	if (s != MB_OK) throw std::runtime_error(std::string(what) + ": " + mb_status_string(s) + " -- " + eb);
}

// The library's public output is XYZW; converted once here to the matrix the
// gate compares (rule 11).
void xyzw_matrix(const float *q, float *m) {
	const float x = q[0], y = q[1], z = q[2], w = q[3];
	m[0] = 1 - 2 * (y * y + z * z);
	m[1] = 2 * (x * y - z * w);
	m[2] = 2 * (x * z + y * w);
	m[3] = 2 * (x * y + z * w);
	m[4] = 1 - 2 * (x * x + z * z);
	m[5] = 2 * (y * z - x * w);
	m[6] = 2 * (x * z - y * w);
	m[7] = 2 * (y * z + x * w);
	m[8] = 1 - 2 * (x * x + y * y);
}

int mbricks_run(int argc, char **argv) {
	if (argc != 7) throw std::runtime_error("mbricks BUNDLE STYLE.mbstyle SECONDS SEED OUTDIR");
	char eb[512] = { 0 };
	const double seconds = std::stod(argv[4]);
	const uint64_t seed = std::stoull(argv[5]);
	mb_runtime_options *opts = nullptr;
	mb_model *model = nullptr;
	mb_style *st = nullptr;
	mb_agent *agent = nullptr;
	mb_command *cmd = nullptr;
	check(mb_runtime_options_create(&opts, eb, sizeof eb), "options", eb);
	check(mb_runtime_options_set_device(opts, MB_DEVICE_CPU, eb, sizeof eb), "device", eb);
	// The desk's 16 hardware threads ran ggml-cpu several times slower than 8 (AGENTS.md).
	const char *mt = std::getenv("MB_THREADS");
	check(mb_runtime_options_set_threads(opts, uint32_t(mt ? std::atoi(mt) : 8), eb, sizeof eb), "threads", eb);
	check(mb_model_load(argv[2], opts, &model, eb, sizeof eb), "model_load", eb);
	check(mb_style_load(model, argv[3], &st, eb, sizeof eb), "style_load", eb);
	check(mb_agent_create(model, &agent, eb, sizeof eb), "agent", eb);
	check(mb_agent_reset(agent, st, eb, sizeof eb), "reset", eb);
	check(mb_command_create(&cmd, eb, sizeof eb), "command", eb);
	check(mb_command_set_style(cmd, st, eb, sizeof eb), "set_style", eb);
	check(mb_command_set_target_speed(cmd, 1.2f, eb, sizeof eb), "speed", eb);
	check(mb_command_set_seed(cmd, seed, eb, sizeof eb), "seed", eb);
	uint32_t J = 0;
	check(mb_model_get_joint_count(model, &J, eb, sizeof eb), "joints", eb);
	const size_t want = size_t(seconds * 30.0);
	std::vector<float> roots, local;
	double heading = 0.0;
	int plans = 0;
	while (roots.size() / 3 < want) {
		heading += 0.6 * std::sin(double(roots.size() / 3) / 30.0 * 0.5);
		const float dx = float(std::sin(heading)), dz = float(std::cos(heading));
		check(mb_command_set_movement_direction(cmd, dx, 0.f, dz, eb, sizeof eb), "dir", eb);
		check(mb_command_set_facing_direction(cmd, dx, 0.f, dz, eb, sizeof eb), "face", eb);
		mb_motion *m = nullptr;
		check(mb_agent_plan(agent, cmd, &m, eb, sizeof eb), "plan", eb);
		uint64_t jn = 0, tn = 0, rn = 0;
		const float *tp = nullptr, *rp = nullptr;
		mb_motion_get_joint_count(m, &jn, eb, sizeof eb);
		mb_motion_get_root_translations(m, &tp, &tn, eb, sizeof eb);
		mb_motion_get_local_rotations_xyzw(m, &rp, &rn, eb, sizeof eb);
		const size_t take = std::min<size_t>(tn / 3, want - roots.size() / 3);
		for (size_t i = 0; i < take; ++i) {
			roots.insert(roots.end(), tp + i * 3, tp + i * 3 + 3);
			for (size_t j = 0; j < jn; ++j) {
				float mat[9];
				xyzw_matrix(rp + (i * jn + j) * 4, mat);
				local.insert(local.end(), mat, mat + 9);
			}
		}
		mb_motion_free(m);
		++plans;
		if (take == 0) throw std::runtime_error("planner returned no frames");
		check(mb_agent_advance(agent, uint32_t(take), eb, sizeof eb), "advance", eb);
	}
	const size_t F = roots.size() / 3;
	std::vector<float> neutral(size_t(J) * 3);
	std::vector<int32_t> parent(J);
	for (uint32_t j = 0; j < J; ++j) {
		check(mb_model_get_joint_parent(model, j, &parent[j], eb, sizeof eb), "parent", eb);
		check(mb_model_get_neutral_joint_position(model, j, &neutral[size_t(j) * 3], &neutral[size_t(j) * 3 + 1],
					  &neutral[size_t(j) * 3 + 2], eb, sizeof eb),
				"neutral", eb);
	}
	std::vector<float> xf(F * J * 12);
	for (size_t f = 0; f < F; ++f)
		for (size_t j = 0; j < J; ++j) {
			float *o = &xf[(f * J + j) * 12];
			std::memcpy(o, &local[(f * J + j) * 9], 9 * sizeof(float));
			const int p = parent[j];
			for (int k = 0; k < 3; ++k)
				o[9 + k] = p < 0 ? roots[f * 3 + size_t(k)] : neutral[j * 3 + size_t(k)] - neutral[size_t(p) * 3 + size_t(k)];
		}
	fs::create_directories(argv[6]);
	write_f32(fs::path(argv[6]) / "transforms.f32", xf);
	std::printf("mbricks %zu frames x %u joints from %d plans, seed %llu\n", F, J, plans, (unsigned long long)seed);
	mb_command_free(cmd);
	mb_agent_free(agent);
	mb_style_free(st);
	mb_model_free(model);
	mb_runtime_options_free(opts);
	return 0;
}

int compare(int argc, char **argv) {
	if (argc != 4 && argc != 5) throw std::runtime_error("compare A.f32 B.f32 [tolerance]");
	const double tol = argc == 5 ? std::stod(argv[4]) : -1.0;
	const auto a = read_f32(argv[2]), b = read_f32(argv[3]);
	if (a.size() != b.size()) {
		std::printf("n %zu vs %zu SIZE MISMATCH\n", a.size(), b.size());
		return 1;
	}
	double mx = 0, ss = 0;
	size_t at = 0, nonfinite = 0;
	for (size_t i = 0; i < a.size(); ++i) {
		const double d = std::fabs(double(a[i]) - double(b[i]));
		if (!std::isfinite(d)) {
			++nonfinite;
			continue;
		}
		ss += d * d;
		if (d > mx) {
			mx = d;
			at = i;
		}
	}
	std::printf("n %zu max_abs %.6g at %zu (%.9g vs %.9g) rms %.6g nonfinite %zu\n", a.size(), mx, at,
			a.empty() ? 0.0 : double(a[at]), b.empty() ? 0.0 : double(b[at]), a.empty() ? 0.0 : std::sqrt(ss / double(a.size())),
			nonfinite);
	if (tol >= 0.0) {
		const bool ok = nonfinite == 0 && mx <= tol;
		std::printf("%s (tolerance %g)\n", ok ? "WITHIN" : "OUTSIDE", tol);
		return ok ? 0 : 1;
	}
	return 0;
}

} // namespace

int main(int argc, char **argv) try {
	const std::string cmd = argc > 1 ? argv[1] : "";
	if (cmd == "embed") return embed(argc, argv);
	if (cmd == "kimodo") return kimodo_run(argc, argv);
	if (cmd == "mbricks") return mbricks_run(argc, argv);
	if (cmd == "compare") return compare(argc, argv);
	std::fprintf(stderr, "usage: motion_native embed|kimodo|mbricks|compare ...\n");
	return 2;
} catch (const std::exception &e) {
	std::fprintf(stderr, "motion_native: %s\n", e.what());
	return 1;
}
