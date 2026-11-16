#!/usr/bin/env python3
"""Per-test status matrix across the _AF / _CPU / _CUDA variants of each suite.

Usage: parity.py <log-dir> [<log-dir> ...]   (logs written by run_tests.py; later
directories override earlier ones). Statuses: P(ass) F(ail) S(kip) C(rash: started but
the process died) -(absent). Prints totals and every test whose status differs.
"""
import os, re, sys
from collections import defaultdict

VARIANTS = ['_AF', '_CPU', '_CUDA']
status, reasons = defaultdict(dict), {}
for d in sys.argv[1:]:
    for f in os.listdir(d):
        m = re.match(r'(.+?)(_AF|_CPU|_CUDA)\.log$', f)
        if not m:
            continue
        base, var = m.groups()
        cur = None
        for line in open(os.path.join(d, f), errors='replace'):
            r = re.match(r'^\[ RUN      \] (\S+)', line)
            if r:
                cur = r.group(1)
                continue
            r = re.match(r'^\[\s+(OK|FAILED|SKIPPED)\s+\] (\S+) \(', line)
            if r:
                status[(base, r.group(2))][var] = {'OK': 'P', 'FAILED': 'F', 'SKIPPED': 'S'}[r.group(1)]
                cur = None
                continue
            if cur and any(k in line for k in ('exception', 'Failure', 'Skipped')):
                reasons.setdefault((base, cur, var), line.strip()[:160])
        if cur:
            status[(base, cur)][var] = 'C'

totals = {v: defaultdict(int) for v in VARIANTS}
for vs in status.values():
    for v in VARIANTS:
        totals[v][vs.get(v, '-')] += 1
print('Totals (P/F/S/C/-):')
for v in VARIANTS:
    t = totals[v]
    print(f"  {v:6} P={t['P']} F={t['F']} S={t['S']} C={t['C']} absent={t['-']}")
print('\nTests whose status differs across variants (AF | CPU | CUDA):')
for (base, test), vs in sorted(status.items()):
    row = [vs.get(v, '-') for v in VARIANTS]
    if len(set(row)) > 1:
        note = next((reasons.get((base, test, v), '') for v in VARIANTS
                     if reasons.get((base, test, v))), '')
        print(f"  {base:28} {test:58} {' | '.join(row)}   {note}")
