#pragma once
// Tiny test helper: CHECK(cond) / CHECK_NEAR(a, b, tol). No dependencies.
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace testing {
inline int &failures() { static int n = 0; return n; }
} // namespace testing

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
      ++testing::failures();                                                     \
    }                                                                            \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                                    \
  do {                                                                           \
    const double va_ = (a), vb_ = (b);                                           \
    if (!(std::fabs(va_ - vb_) <= (tol))) {                                      \
      std::fprintf(stderr, "%s:%d: CHECK_NEAR failed: %s=%g vs %s=%g (tol %g)\n", \
                   __FILE__, __LINE__, #a, va_, #b, vb_, double(tol));           \
      ++testing::failures();                                                     \
    }                                                                            \
  } while (0)

#define TEST_MAIN_END()                                                          \
  do {                                                                           \
    if (testing::failures()) {                                                   \
      std::fprintf(stderr, "%d check(s) failed\n", testing::failures());         \
      return 1;                                                                  \
    }                                                                            \
    std::printf("ok\n");                                                         \
    return 0;                                                                    \
  } while (0)
