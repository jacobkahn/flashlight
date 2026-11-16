#!/usr/bin/env python3
"""Run every ctest-registered Flashlight test binary and summarize the results.

Usage: run_tests.py <build-dir> <log-dir>

- Binaries come from <build-dir>/flashlight/fl/test/CTestTestfile.cmake (gtest suite
  names often differ from binary names, so don't derive binaries from `ctest -N`).
- The backend variants of one suite (Foo, Foo_CPU, Foo_CUDA, Foo_AF) run serially,
  because some tests write fixed file names under /tmp; different suites run in
  parallel and are spread round-robin over GPUs.
- Environment knobs: ONLY=<regex> selects binaries; JOBS (default 16) parallel groups;
  NUM_GPUS (default 8); OMP_THREADS (default 16, see DEVELOPMENT.md); TIMEOUT seconds
  per binary (default 1800).

Writes one log per binary plus summary.json, and prints problem binaries.
"""
import json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

build, outdir = sys.argv[1], sys.argv[2]
bindir = os.path.join(build, 'flashlight', 'fl', 'test')
os.makedirs(outdir, exist_ok=True)
jobs = int(os.environ.get('JOBS', 16))
gpus = int(os.environ.get('NUM_GPUS', 8))
omp = os.environ.get('OMP_THREADS', '16')
timeout = int(os.environ.get('TIMEOUT', 1800))

ctf = open(os.path.join(bindir, 'CTestTestfile.cmake')).read()
names = {os.path.basename(m.group(1))
         for m in re.finditer(r'add_test\(\S+ "([^"]+)"', ctf)}
bins = sorted(n for n in names if os.path.isfile(os.path.join(bindir, n)))
if os.environ.get('ONLY'):
    bins = [n for n in bins if re.search(os.environ['ONLY'], n)]
groups = {}
for b in bins:
    groups.setdefault(re.sub(r'_(CPU|CUDA|AF)$', '', b), []).append(b)

def run_group(item):
    idx, (_, members) = item
    results = []
    for b in members:
        env = dict(os.environ, OMP_NUM_THREADS=omp,
                   CUDA_VISIBLE_DEVICES=str(idx % gpus))
        try:
            p = subprocess.run([os.path.join(bindir, b)], cwd=bindir, env=env,
                               capture_output=True, text=True, timeout=timeout)
            out, rc = p.stdout + p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode(errors='replace') if isinstance(e.stdout, bytes) else (e.stdout or '')
            rc = 'TIMEOUT'
        open(os.path.join(outdir, b + '.log'), 'w').write(out)
        passed = re.search(r'\[  PASSED  \] (\d+) test', out)
        failed = sorted(set(re.findall(r'^\[  FAILED  \] (\S+?)(?: \(|$)', out, re.M)))
        results.append(dict(bin=b, rc=rc, passed=int(passed.group(1)) if passed else 0,
                            failed=failed))
    return results

with ThreadPoolExecutor(max_workers=jobs) as ex:
    results = [r for g in ex.map(run_group, enumerate(sorted(groups.items()))) for r in g]
json.dump(results, open(os.path.join(outdir, 'summary.json'), 'w'), indent=1)
bad = [r for r in results if r['failed'] or r['rc'] != 0]
print(f"binaries={len(results)} passed_tests={sum(r['passed'] for r in results)} "
      f"problem_binaries={len(bad)}")
for r in bad:
    print(r['bin'], 'rc=', r['rc'], 'passed=', r['passed'], 'failed=', r['failed'][:8])
