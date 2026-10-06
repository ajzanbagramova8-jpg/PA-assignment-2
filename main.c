#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PI 3.14159265358979323846

static double rnd(unsigned long long *s) {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    return (double)(*s >> 11) * (1.0 / 9007199254740992.0);
}

void p1(void) {
    printf("\n Problem 1\n");
    const long long N = 10000000;
    double *A = malloc(N * sizeof(double));
    double *B = malloc(N * sizeof(double));
    double *C = malloc(N * sizeof(double));
    double *S = malloc(N * sizeof(double));
    if (!A || !B || !C || !S) {
        printf("Memory allocation failed\n");
        return;
    }

    #pragma omp parallel for
    for (long long i = 0; i < N; i++) {
        A[i] = i * 0.5;
        B[i] = i * 1.5;
    }

    double t = omp_get_wtime();
    for (long long i = 0; i < N; i++) S[i] = A[i] + B[i];
    double ts = omp_get_wtime() - t;

    t = omp_get_wtime();
    #pragma omp parallel for
    for (long long i = 0; i < N; i++) C[i] = A[i] + B[i];
    double tp = omp_get_wtime() - t;

    printf("N = %lld, threads = %d\n", N, omp_get_max_threads());
    printf("First 5 results:\n");
    for (int i = 0; i < 5; i++)
        printf("  C[%d] = %.1f (%.1f + %.1f)\n", i, C[i], A[i], B[i]);
    printf("Last 5 results:\n");
    for (long long i = N - 5; i < N; i++)
        printf("  C[%lld] = %.1f (%.1f + %.1f)\n", i, C[i], A[i], B[i]);

    int ok = 1;
    for (long long i = 0; i < N; i++)
        if (C[i] != S[i]) { ok = 0; break; }
    printf("Verification: %s\n", ok ? "OK" : "MISMATCH");

    printf("Serial time:   %.6f s\n", ts);
    printf("Parallel time: %.6f s\n", tp);
    printf("Speed-up:      %.6fx\n", ts / tp);

    free(A); free(B); free(C); free(S);
}

void p2(void) {
    printf("\n Problem 2:\n");
    const long long N = 200000000;
    const double h = 1.0 / N;
    double s = 0.0;

    double t = omp_get_wtime();
    #pragma omp parallel for reduction(+ : s)
    for (long long i = 0; i < N; i++) {
        double x = (i + 0.5) * h;
        s += 4.0 / (1.0 + x * x);
    }
    double pi = s * h;
    t = omp_get_wtime() - t;

    printf("N = %lld, threads = %d\n", N, omp_get_max_threads());
    printf("Approximation of pi: %.15f\n", pi);
    printf("Reference pi:        %.15f\n", PI);
    printf("Absolute error:      %.3e\n", fabs(pi - PI));
    printf("Time: %.6f s\n", t);
}

int prime(long long n) {
    if (n < 2) return 0;
    if (n % 2 == 0) return n == 2;
    for (long long d = 3; d * d <= n; d += 2)
        if (n % d == 0) return 0;
    return 1;
}

void p3(void) {
    printf("\nProblem 3\n");
    const long long N = 2000000;

    long long cc = 0;
    double t = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 1000)
    for (long long n = 2; n <= N; n++) {
        if (prime(n)) {
            #pragma omp critical
            cc++;
        }
    }
    double tc = omp_get_wtime() - t;

    long long ca = 0;
    t = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 1000)
    for (long long n = 2; n <= N; n++) {
        if (prime(n)) {
            #pragma omp atomic
            ca++;
        }
    }
    double ta = omp_get_wtime() - t;

    printf("Number of primes in [2, %lld]:\n", N);
    printf("  critical: %lld  (%.6f s)\n", cc, tc);
    printf("  atomic:   %lld  (%.6f s)\n", ca, ta);
}

void p4(void) {
    printf("\nProblem 4: \n");
    const long long M = 100000000;
    long long *lc = calloc(omp_get_max_threads(), sizeof(long long));
    long long in = 0, tot = 0;
    int nu = 1;

    double t = omp_get_wtime();
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int nt = omp_get_num_threads();

        #pragma omp single
        nu = nt;

        long long ch = M / nt;
        long long mp = (id == nt - 1) ? M - ch * (nt - 1) : ch;

        unsigned long long s = (unsigned long long)(id + 1) * 0x9E3779B97F4A7C15ULL;
        long long c = 0;
        for (long long i = 0; i < mp; i++) {
            double x = rnd(&s), y = rnd(&s);
            if (x * x + y * y <= 1.0) c++;
        }
        lc[id] = c;

        #pragma omp barrier

        #pragma omp master
        {
            for (int k = 0; k < nt; k++) in += lc[k];
            tot = M;
        }
    }
    t = omp_get_wtime() - t;

    double pi = 4.0 * (double)in / (double)tot;
    printf("Threads: %d, points: %lld\n", nu, tot);
    printf("Points inside circle: %lld\n", in);
    printf("Estimated pi: %.10f  (error %.10f)\n", pi, fabs(pi - PI));
    printf("Time: %.6f s\n", t);

    free(lc);
}

long long fs(int n) { return n < 2 ? n : fs(n - 1) + fs(n - 2); }

long long ft(int n, int cut) {
    if (n <= cut) return fs(n);
    long long x, y;
    #pragma omp task shared(x)
    x = ft(n - 1, cut);
    #pragma omp task shared(y)
    y = ft(n - 2, cut);
    #pragma omp taskwait
    return x + y;
}

void p5(void) {
    printf("\nProblem 5:\n");
    const int n = 40;
    const int cut = 25;
    long long r = 0;

    double t = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        r = ft(n, cut);
    }
    double tp = omp_get_wtime() - t;

    t = omp_get_wtime();
    long long ref = fs(n);
    double ts = omp_get_wtime() - t;

    printf("F(%d) = %lld%s\n", n, r, r == ref ? "  (verified)" : "  (MISMATCH)");
    printf("Cutoff = %d, threads = %d\n", cut, omp_get_max_threads());
    printf("Parallel (tasks) time: %.6f s\n", tp);
    printf("Sequential time:       %.6f s\n", ts);
    printf("Speed-up:              %.6fx\n", ts / tp);
}

int main(void) {
    printf("Max threads available: %d\n", omp_get_max_threads());
    p1();
    p2();
    p3();
    p4();
    p5();
    return 0;
}