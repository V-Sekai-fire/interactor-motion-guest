#!/usr/bin/env python3
"""The route table's input paths must equal the engine's controller profile, in both directions.

    python3 check_profile.py <engine checkout> <routes paths file>
    python3 check_profile.py --self-test

The profile is the extension source under modules/openxr/extensions/ that registers the view and
d-pad clicks; per-hand paths are registered once and expanded to both hands.
"""

import re
import sys
from pathlib import Path

MARKERS = ('"/user/hand/left/input/view/click"', '"/user/hand/left/input/dpad_up/click"')
HANDS = ("/user/hand/left", "/user/hand/right")


def profile_source(engine):
    found = [p for p in sorted(Path(engine, "modules/openxr/extensions").glob("*.cpp"))
             if all(m in p.read_text(encoding="utf-8", errors="replace") for m in MARKERS)]
    if len(found) != 1:
        raise SystemExit("FAIL: expected one profile source with the view and d-pad clicks, found %d" % len(found))
    return found[0].read_text(encoding="utf-8")


def engine_paths(text):
    paths = set()
    for call in re.finditer(r"register_io_path\(profile_path,\s*\"[^\"]*\",\s*([^,]+),\s*([^,]+?),", text):
        expr = call.group(2).strip()
        per_hand = re.fullmatch(r'user_path\s*\+\s*"([^"]+)"', expr)
        literal = re.fullmatch(r'"([^"]+)"', expr)
        if per_hand:
            paths.update(h + per_hand.group(1) for h in HANDS)
        elif literal:
            paths.add(literal.group(1))
        else:
            raise SystemExit("FAIL: an input path the parser cannot read: %s" % expr)
    return paths


def compare(engine, routes):
    problems = ["missing from the route table: " + p for p in sorted(engine - routes)]
    problems += ["not in the engine's profile: " + p for p in sorted(routes - engine)]
    return problems


def self_test():
    sample = '''
    for (const String user_path : { "/user/hand/left", "/user/hand/right" }) {
        p->register_io_path(profile_path, "Grip pose", user_path, user_path + "/input/grip/pose", "", X);
    }
    p->register_io_path(profile_path, "A click", "/user/hand/right", "/user/hand/right/input/a/click", "", X);
    '''
    engine = engine_paths(sample)
    routes = {"/user/hand/left/input/grip/pose", "/user/hand/right/input/grip/pose", "/user/hand/right/input/a/click"}
    cases = [
        ("the same set passes", compare(engine, routes) == []),
        ("an engine input the table lacks fails", compare(engine | {"/user/hand/left/input/new/click"}, routes) != []),
        ("a table input the engine lacks fails", compare(engine, routes | {"/user/hand/left/input/gone/click"}) != []),
        ("both hands come from one per-hand registration", len(engine) == 3),
    ]
    for name, ok in cases:
        print("%s %s" % ("ok  " if ok else "FAIL", name))
    return 0 if all(ok for _, ok in cases) else 1


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    if len(argv) != 2:
        print(__doc__)
        return 2
    engine = engine_paths(profile_source(argv[0]))
    routes = {line.strip() for line in Path(argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()}
    problems = compare(engine, routes)
    for p in problems:
        print("FAIL " + p)
    print("profile: %d engine inputs, %d routed, %d differences" % (len(engine), len(routes), len(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
