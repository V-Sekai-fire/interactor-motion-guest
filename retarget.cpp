#include "retarget.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace motion {

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kProbe = 30.f * kPi / 180.f; // the rotation each axis is tried with

const char *kRoleNames[ROLE_COUNT] = { "LeftUpperArm", "RightUpperArm", "LeftLowerArm", "RightLowerArm",
	"LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg" };

float dot3(const float *a, const float *b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void cross3(const float *a, const float *b, float *o) {
	o[0] = a[1] * b[2] - a[2] * b[1];
	o[1] = a[2] * b[0] - a[0] * b[2];
	o[2] = a[0] * b[1] - a[1] * b[0];
}

bool normalize3(float *v) {
	const float n = std::sqrt(dot3(v, v));
	if (!(n > 1e-12f)) {
		return false;
	}
	v[0] /= n;
	v[1] /= n;
	v[2] /= n;
	return true;
}

void mul3(const float *a, const float *b, float *o) { // o = a b, row-major
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			o[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
		}
	}
}

void apply3(const float *m, const float *v, float *o) {
	for (int i = 0; i < 3; ++i) {
		o[i] = m[i * 3] * v[0] + m[i * 3 + 1] * v[1] + m[i * 3 + 2] * v[2];
	}
}

// Gram-Schmidt on the columns (x, then y, then z = x cross y): a rotation even
// if the asset's frame carries scale.
void orthonormalize_columns(float *m) {
	float x[3] = { m[0], m[3], m[6] }, y[3] = { m[1], m[4], m[7] }, z[3];
	normalize3(x);
	const float d = dot3(x, y);
	for (int i = 0; i < 3; ++i) {
		y[i] -= d * x[i];
	}
	normalize3(y);
	cross3(x, y, z);
	for (int i = 0; i < 3; ++i) {
		m[i * 3] = x[i];
		m[i * 3 + 1] = y[i];
		m[i * 3 + 2] = z[i];
	}
}

} // namespace

const char *role_name(int role) {
	return role >= 0 && role < ROLE_COUNT ? kRoleNames[role] : "?";
}

bool role_is_hinge(int role) {
	return role == L_LOWER_ARM || role == R_LOWER_ARM || role == L_LOWER_LEG || role == R_LOWER_LEG;
}

int Skeleton::find(const std::string &name) const {
	for (size_t i = 0; i < names.size(); ++i) {
		if (names[i] == name) {
			return int(i);
		}
	}
	return -1;
}

void axis_angle_matrix(const float axis[3], float angle, float out[9]) {
	const float c = std::cos(angle), s = std::sin(angle), t = 1.f - c;
	const float x = axis[0], y = axis[1], z = axis[2];
	out[0] = t * x * x + c;
	out[1] = t * x * y - s * z;
	out[2] = t * x * z + s * y;
	out[3] = t * x * y + s * z;
	out[4] = t * y * y + c;
	out[5] = t * y * z - s * x;
	out[6] = t * x * z - s * y;
	out[7] = t * y * z + s * x;
	out[8] = t * z * z + c;
}

float angle_about(const float r[9], const float axis[3]) {
	// u: the unit vector perpendicular to the axis built from the least
	// aligned basis vector.
	const int k = std::fabs(axis[0]) < 0.6f ? 0 : (std::fabs(axis[1]) < 0.6f ? 1 : 2);
	float e[3] = { 0, 0, 0 };
	e[k] = 1.f;
	float u[3];
	const float d = dot3(e, axis);
	for (int i = 0; i < 3; ++i) {
		u[i] = e[i] - d * axis[i];
	}
	normalize3(u);
	float v[3];
	apply3(r, u, v);
	const float va = dot3(v, axis);
	for (int i = 0; i < 3; ++i) {
		v[i] -= va * axis[i];
	}
	float c[3];
	cross3(u, v, c);
	return std::atan2(dot3(c, axis), dot3(u, v));
}

Rig measure(const Skeleton &in, const int joints[ROLE_COUNT], const int ends[ROLE_COUNT], int left_hip,
		int right_hip) {
	Rig rig;
	rig.skel = in;
	const int nj = int(in.names.size());
	if (int(in.parent.size()) != nj || int(in.rot.size()) != nj * 9 || int(in.pos.size()) != nj * 3) {
		rig.error = "skeleton arrays do not match its joint count";
		return rig;
	}
	for (int j = 0; j < nj; ++j) {
		orthonormalize_columns(&rig.skel.rot[size_t(j) * 9]);
	}
	if (left_hip < 0 || right_hip < 0 || left_hip >= nj || right_hip >= nj) {
		rig.error = "no hips to take forward from";
		return rig;
	}
	const float up[3] = { 0, 1, 0 };
	float lr[3];
	for (int i = 0; i < 3; ++i) {
		lr[i] = in.pos[size_t(right_hip) * 3 + i] - in.pos[size_t(left_hip) * 3 + i];
	}
	cross3(up, lr, rig.forward);
	rig.forward[1] = 0.f;
	if (!normalize3(rig.forward)) {
		rig.error = "hips are not side by side";
		return rig;
	}
	for (int r = 0; r < ROLE_COUNT; ++r) {
		Axis &a = rig.axis[r];
		a.joint = joints[r];
		a.end = ends[r];
		if (a.joint < 0 || a.end < 0 || a.joint >= nj || a.end >= nj) {
			rig.error = std::string("role ") + role_name(r) + " has no joint or no end";
			return rig;
		}
		const float *R = &rig.skel.rot[size_t(a.joint) * 9];
		const float *p = &in.pos[size_t(a.joint) * 3];
		const float *e = &in.pos[size_t(a.end) * 3];
		const float rel[3] = { e[0] - p[0], e[1] - p[1], e[2] - p[2] };
		// The limb bends forward, except the knee, which bends back.
		const float bend = (r == L_LOWER_LEG || r == R_LOWER_LEG) ? -1.f : 1.f;
		int best = -1;
		float best_fwd = 0.f;
		for (int k = 0; k < 3; ++k) {
			const float ax[3] = { R[k], R[3 + k], R[6 + k] }; // local axis k in world
			float rot[9], moved[3];
			axis_angle_matrix(ax, kProbe, rot);
			apply3(rot, rel, moved);
			const float dsp[3] = { moved[0] - rel[0], moved[1] - rel[1], moved[2] - rel[2] };
			a.travel_mm[k] = 1000.f * std::sqrt(dot3(dsp, dsp));
			a.forward_mm[k] = 1000.f * dot3(dsp, rig.forward);
			a.up_mm[k] = 1000.f * dsp[1];
			if (best < 0 || std::fabs(a.forward_mm[k]) > std::fabs(best_fwd)) {
				best = k;
				best_fwd = a.forward_mm[k];
			}
		}
		// A segment that already points forward (G1's forearm at rest) moves its
		// end forward little about any axis; its flexion lifts the end instead.
		// When the runner-up is within 20% of the best, the one that also lifts
		// the end more is the flexion axis.
		for (int k = 0; k < 3; ++k) {
			if (k != best && std::fabs(a.forward_mm[k]) >= 0.8f * std::fabs(best_fwd)) {
				a.tie = true;
				if (std::fabs(a.up_mm[k]) > std::fabs(a.up_mm[best])) {
					best = k;
					best_fwd = a.forward_mm[k];
				}
			}
		}
		a.local = best;
		a.sign = (best_fwd * bend) >= 0.f ? 1.f : -1.f;
	}
	return rig;
}

void retarget_begin(Retargeted &out, int frames) {
	out.frames = frames;
	out.local.assign(size_t(frames) * ROLE_COUNT * 12, 0.f);
	out.target_deg.assign(size_t(frames) * ROLE_COUNT, 0.f);
	out.source_deg.assign(size_t(frames) * ROLE_COUNT, 0.f);
}

Retargeted retarget(const Rig &src, const float *source_local, int frames, int source_joints, const Rig &dst) {
	Retargeted out;
	retarget_begin(out, frames);
	retarget_range(src, source_local, 0, frames, source_joints, dst, out);
	return out;
}

void retarget_range(const Rig &src, const float *source_local, int f0, int f1, int source_joints, const Rig &dst,
		Retargeted &out) {
	for (int r = 0; r < ROLE_COUNT; ++r) {
		const Axis &sa = src.axis[r];
		const Axis &da = dst.axis[r];
		float s_axis[3] = { 0, 0, 0 };
		s_axis[sa.local] = sa.sign;
		float d_axis[3] = { 0, 0, 0 };
		d_axis[da.local] = da.sign;
		// The avatar bone's rest transform relative to its parent.
		float rest[9], trans[3];
		const int b = da.joint;
		const int pb = dst.skel.parent[size_t(b)];
		if (!dst.skel.local.empty()) {
			const float *l = &dst.skel.local[size_t(b) * 12];
			for (int i = 0; i < 9; ++i) {
				rest[i] = l[i];
			}
			trans[0] = l[9];
			trans[1] = l[10];
			trans[2] = l[11];
		} else if (pb < 0) {
			for (int i = 0; i < 9; ++i) {
				rest[i] = dst.skel.rot[size_t(b) * 9 + i];
			}
			for (int i = 0; i < 3; ++i) {
				trans[i] = dst.skel.pos[size_t(b) * 3 + i];
			}
		} else {
			const float *P = &dst.skel.rot[size_t(pb) * 9];
			float pt[9];
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					pt[i * 3 + j] = P[j * 3 + i];
				}
			}
			mul3(pt, &dst.skel.rot[size_t(b) * 9], rest);
			const float d[3] = { dst.skel.pos[size_t(b) * 3] - dst.skel.pos[size_t(pb) * 3],
				dst.skel.pos[size_t(b) * 3 + 1] - dst.skel.pos[size_t(pb) * 3 + 1],
				dst.skel.pos[size_t(b) * 3 + 2] - dst.skel.pos[size_t(pb) * 3 + 2] };
			apply3(pt, d, trans);
		}
		for (int f = f0; f < f1; ++f) {
			const float *L = source_local + (size_t(f) * source_joints + size_t(sa.joint)) * 9;
			const float src_angle = angle_about(L, s_axis);
			const float angle = role_is_hinge(r) ? std::fabs(src_angle) : src_angle;
			float flex[9], m[9];
			axis_angle_matrix(d_axis, angle, flex);
			mul3(rest, flex, m);
			float *o = &out.local[(size_t(f) * ROLE_COUNT + size_t(r)) * 12];
			for (int i = 0; i < 9; ++i) {
				o[i] = m[i];
			}
			o[9] = trans[0];
			o[10] = trans[1];
			o[11] = trans[2];
			out.source_deg[size_t(f) * ROLE_COUNT + size_t(r)] = src_angle * 180.f / kPi;
			out.target_deg[size_t(f) * ROLE_COUNT + size_t(r)] = angle * 180.f / kPi;
		}
	}
}

std::string ranges_text(const std::string &clip, const Retargeted &rt) {
	std::string s;
	char buf[256];
	for (int r = 0; r < ROLE_COUNT; ++r) {
		float tmin = 1e30f, tmax = -1e30f, smin = 1e30f, smax = -1e30f;
		for (int f = 0; f < rt.frames; ++f) {
			const float t = rt.target_deg[size_t(f) * ROLE_COUNT + size_t(r)];
			const float v = rt.source_deg[size_t(f) * ROLE_COUNT + size_t(r)];
			tmin = std::min(tmin, t);
			tmax = std::max(tmax, t);
			smin = std::min(smin, v);
			smax = std::max(smax, v);
		}
		std::snprintf(buf, sizeof buf, "%-28s %-14s %s  avatar %7.1f .. %7.1f deg  source %7.1f .. %7.1f deg  span %6.1f\n",
				clip.c_str(), role_name(r), role_is_hinge(r) ? "hinge |a|" : "signed   ", tmin, tmax, smin, smax,
				tmax - tmin);
		s += buf;
	}
	return s;
}

std::string axes_text(const std::string &what, const Rig &rig) {
	std::string s;
	char buf[512];
	std::snprintf(buf, sizeof buf, "%s forward (%.3f, %.3f, %.3f)\n", what.c_str(), rig.forward[0], rig.forward[1],
			rig.forward[2]);
	s += buf;
	static const char *xyz = "XYZ";
	for (int r = 0; r < ROLE_COUNT; ++r) {
		const Axis &a = rig.axis[r];
		std::snprintf(buf, sizeof buf,
				"%s %-14s joint %-28s end %-24s axis %c%c%s | 30 deg about local X, Y, Z moves the end %6.1f %6.1f %6.1f mm (forward %+6.1f %+6.1f %+6.1f, up %+6.1f %+6.1f %+6.1f)\n",
				what.c_str(), role_name(r), a.joint >= 0 ? rig.skel.names[size_t(a.joint)].c_str() : "?",
				a.end >= 0 ? rig.skel.names[size_t(a.end)].c_str() : "?", a.sign > 0 ? '+' : '-',
				a.local >= 0 ? xyz[a.local] : '?', a.tie ? " (tie: lifts)" : "             ", a.travel_mm[0], a.travel_mm[1], a.travel_mm[2], a.forward_mm[0],
				a.forward_mm[1], a.forward_mm[2], a.up_mm[0], a.up_mm[1], a.up_mm[2]);
		s += buf;
	}
	return s;
}

} // namespace motion
