# HyperplaneRegionEnum

The `HyperplaneRegionEnum` Python package implements a (growing) number of algorithms to enumerate the regions in a hyperplane arrangement. Additionally, this package has the following key features:

* **Per Region Testing and Early Termination:** During enumeration, user-specified code can be evaluated to test each region, and terminate the enumeration early. This feature makes it useful for a number of applications in Neural Network (NN) verification (see e.g. [FastBATLLNN](https://github.com/jferlez/FastBATLLNN)).
* **Parallelism:** This package contains parallel implementations of all algorithms, implemented using the [charm4py](https://charm4py.readthedocs.io/en/latest/) parallelism abstraction framework. Since [charm4py](https://charm4py.readthedocs.io/en/latest/) runs on top of MPI, these algorithms can be scaled up to multi-computer clusters with ease.

_Please contact [jferlez@uci.edu](mailto:jferlez@uci.edu) with any questions/bug reports._

## 1) Prerequisites

This package is in Python, and it depends on the following packages/libraries (and their dependencies):

* [charm4py](https://charm4py.readthedocs.io/en/latest/) (Python/Library)
* [glpk](https://www.gnu.org/software/glpk/) (Library)
* [scikit-glpk](https://github.com/mckib2/scikit-glpk) (Python)
* [cylp](https://github.com/coin-or/CyLP) (Python) / [Clp](https://github.com/coin-or/Clp) (Library)
* [cvxopt](https://cvxopt.org) (Python)
* [numba](https://numba.pydata.org) (Python)
* [pycddlib](https://pycddlib.readthedocs.io/en/latest/) (Python/Library)

These dependencies can be burdensome to install, but they are pre-installed and configured in the Docker image associated with [FastBATLLNN](https://github.com/jferlez/FastBATLLNN); please see that repository for Docker instructions.

> **Note:** [charm4py](https://charm4py.readthedocs.io/en/latest/) is required, so this package must be used within an appropriately invoked Python interpreter: see the [charm4py documentation](https://charm4py.readthedocs.io/en/latest/) for more instructions. You can test your configuration by running the included example from a shell as follows:
> ```Bash
> charmrun +p1 example.py
> ```
> which should produce output that looks roughly like this:
> ```
> Running as 4 OS processes: /opt/local/bin/python3.11 example.py
> charmrun> mpirun -np 4 /opt/local/bin/python3.11 example.py
> Charm++> Running on MPI library: Open MPI v4.1.4, package: Open MPI root@ Distribution, ident: 4.1.4, repo rev: v4.1.4, May 26, 2022 (MPI standard: 3.1)
> Charm++> Level of thread support used: MPI_THREAD_SINGLE (desired: MPI_THREAD_SINGLE)
> Charm++> Running in non-SMP mode: 4 processes (PEs)
> Converse/Charm++ Commit ID: v7.1.0-devel-196-gf9f1f096f
> Isomalloc> Synchronized global address space.
> CharmLB> Load balancer assumes all CPUs are same.
> Charm4py> Running Charm4py version 1.0 on Python 3.11.4 (CPython). Using 'cython' interface to access Charm++
> Charm++> Running on 1 hosts (1 sockets x 4 cores x 2 PUs = 8-way SMP)
> Charm++> cpu topology info is gathered in 0.004 seconds.
> CharmLB> Load balancing instrumentation for communication is off.
> 
> Total number of regions: 4
> 
> [Partition 0][Node 0] End of program
> ```
> That is, `example.py` enumerates an arrangement in $\mathbb{R}^2$ with 4 regions.

## 2) Basic Usage

This section describes code that must be run in a Python interpreter invoked by `charmrun` or similar; see Section [1) Prerequisites](#1-prerequisites). Note that the following sections closely follow the code in the included `example.py`.

### Initialization of an Enumerator
The user-facing interface to the enumerator is a [Chare](https://charm4py.readthedocs.io/en/latest/) abstraction called `posetFastCharm.Poset`. It can be created according the syntax of [Chare](https://charm4py.readthedocs.io/en/latest/) creation as in the following example from `example.py`:

```Python
enumerator = Chare(posetFastCharm.Poset, \
                    args=[ \
                            # PE Specification \
                            # (None => use all PEs)
                            None, \
                            # Constructor for individual region storage instances \
                            # (None => posetFastCharm.PosetNode)
                            None, \
                            # localVarGroup for evaluating region tests \
                            # (None => posetFastCharm.localVar)
                            None, \
                            # Chare to obtain region successors in poset \
                            # (None => posetFastCharm.successorWorker) \
                            None, \
                            # Execute region tests on poset workers (True) or hash workers (False)
                            False, \
                            # Specification for chaining DistributedHash Chares
                            []
                        ], \
                    onPE=0 \
                )
charm.awaitCreation(enumerator)
# Perform initialization outside the constructor:
enumerator.init(awaitable=True).get()
```
> **NOTE:** As required by charm4py, arguments are passed to the Chare constructor via the argument `args`, and specified with a fixed-length list. **_The list above is the "minimal" list of arguments, which invokes default values for all parameters._**

> **NOTE:** The `Chare` constructor returns immediately, so the call to `charm.awaitCreation` is necessary to block until the Chare is actually initialized/created.

> **NOTE:** The `init` method must be called, and its execution completed via `awaitable=True`, before further interaction with the enumerator.

Non-default options to the Chare constructor are covered in Section [3) Advanced Usage](#3-advanced-usage).

### Specifying a Hyperplane Arrangement
`HyperplaneRegionEnum` enumerates all of the regions in a hyperplane arrangement of $\mathbb{R}^d$ that intersect a closed, convex polytope region, $\mathcal{R}$. To enumerate these regions, three quantities are required:

**_(i)_ Hyperplanes:** The hyperplanes themselves are specified according to matrix equation with each row specifying a hyperplane, i.e.:
$$
A x + b = 0
$$
where `A` is a Numpy array of shape $(N, d)$ and `b` is a Numpy array of shape $(N, 1)$.

**_(ii)_ Constraint Region:** The region to search, $\mathcal{R}$, is specified by a linear matrix inequality:
$$
A^f x \geq b^f
$$
where `fA`=$A^f$ is a Numpy array of shape $(M, d)$ and `fb`=$b^f$ is a Numpy array of shape $(M, 1)$.

**_(iii)_ Initial, Interior Region Point:** The user must supply a point $p \in \mathbb{R}^d$ that is _interior_ to one region of the hyperplane arrangement and also contained in $\mathcal{R}$. `pt`=$p$ should be a Numpy array of size $(d, 1)$.

With this information, the enumerator is initialized for the corresponding enumeration problem as follows:
```Python
# Supply hyperplanes to enumerator (.get() call waits for completion):
enumerator.initialize(
        # Hyperplane specification:
        [[A, b]], \
        # Initial region interior point:
        pt, \
        # Fixed region constraints:
        fA, \
        fb, \
        awaitable=True \
    ).get()
# Shift and pre-filter hyperplanes for intersection with fixed constraints
enumerator.setConstraint(0, prefilter=True, awaitable=True).get()
```

> **WARNING:** Each `initialize` call must be followed by a `setConstraint` call.

> **WARNING:** The `awaitable=True` and `.get()` block the main code execution until each corresponding method call completes. These calls must complete, and in order, before proceeding.

> **NOTE:** Subsequent calls to `initialize`/`setConstraint` will replace any enumeration problem configured before.

Further documentation about the `setConstraint` method and its parameters can be found in Section [3) Advanced Usage](#3-advanced-usage).

### Performing the Enumeration
At this point, the enumerator has been configured with an enumeration problem. So the enumeration can be started by calling:
```Python
retVal = enumerator.populatePoset(ret=True).get()
```

> **NOTE:** `populatePoset` should be called with `ret=True` or `awaitable=True` so that the main code can block until enumeration completes.

> **NOTE:** When called with `ret=True`, `populatePoset` returns `True` if user-configured test code returns `True` for all enumerated regions, and `False` if the user-configured test code returns `False` on at one region. If the test code returns `False` on a region, then enumeration is terminated early (i.e. short-circuit evaluation).

> **NOTE:** With default settings `retVal` is always `True` after enumeration, and hence all regions are enumerated.

### Collecting the Results

The enumerator instance can be queried for a Python `defaultdict` containing the results of the enumeration, including the number of regions enumerated and the total number of Linear Programs (LPs) used:

```Python
stats = enumerator.getStats(ret=True).get()
```

## 3) Advanced Usage
Under construction...