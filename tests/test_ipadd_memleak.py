import os
import resource # TODO: Add non-*nix solution
import random
import gc
import sys
from cvxopt import spmatrix

N = 1000
M = 1000
IMAX = 1000
JMAX = 1000

if __name__ == "__main__":
    mem_before = 0
    mem_after = 0

    indices = [(i, j) for i in range(N) for j in range(M)]
    A = spmatrix([random.gauss(0, 1) + random.gauss(0, 1) * 1j for i in range(N * M)], [i for i, j in indices], [j for i, j in indices], (N, M))

    del indices
    gc.collect()

    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory usage before operations: {}".format(mem_before))

    for _ in range(IMAX):
        values = [random.gauss(0, 1) + random.gauss(0, 1) * 1j for i in range(JMAX)]
        indices = [(random.randrange(M), random.randrange(N)) for i in range(JMAX)]
        indices.sort()

        A.ipset(values, [i for i, j in indices], [j for i, j in indices])

    for _ in range(IMAX):
        indices = [(random.randrange(M), random.randrange(N)) for i in range(JMAX)]
        indices.sort()

        A.ipset(random.gauss(0, 1) + random.gauss(0, 1) * 1j, [i for i, j in indices], [j for i, j in indices])

    for _ in range(IMAX):
        A.ipset(random.gauss(0, 1) + random.gauss(0, 1) * 1j, random.randrange(M), random.randrange(N))

    for _ in range(IMAX):
        values = [random.gauss(0, 1) + random.gauss(0, 1) * 1j for i in range(JMAX)]
        indices = [(random.randrange(M), random.randrange(N)) for i in range(JMAX)]
        indices.sort()

        A.ipadd(values, [i for i, j in indices], [j for i, j in indices])

    for _ in range(IMAX):
        indices = [(random.randrange(M), random.randrange(N)) for i in range(JMAX)]
        indices.sort()

        A.ipadd(random.gauss(0, 1) + random.gauss(0, 1) * 1j, [i for i, j in indices], [j for i, j in indices])

    for _ in range(IMAX):
        A.ipadd(random.gauss(0, 1) + random.gauss(0, 1) * 1j, random.randrange(M), random.randrange(N))

    del values
    del indices
    gc.collect()

    mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory usage after operations: {}".format(mem_after))

    if mem_before != mem_after:
        print("Possible memory leak in ipset/ipadd detected")
        exit(-1)
    else:
        print("No memory leak detected")
