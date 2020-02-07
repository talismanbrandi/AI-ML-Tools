#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_rng.h>

int
main (void)
{
  int i;
  const size_t N = 1000;      /* number of points to interpolate */
  const size_t nx = 101;       /* x grid points */
  double xi, yi, x[nx], y[nx];

  for (i = 0; i < nx; i++) {
    x[i] = i * 2. * M_PI/(nx - 1.);
    y[i] = sin(x[i])*cos(5.*x[i]);
    printf ("%g %g\n", x[i], y[i]);
  }
  gsl_interp_accel *acc = gsl_interp_accel_alloc ();
  gsl_spline *spline = gsl_spline_alloc (gsl_interp_cspline, nx);

  gsl_spline_init (spline, x, y, nx);
    
  /* random number generator for generating output */
  const gsl_rng_type * Tr;
  gsl_rng * r;
  gsl_rng_env_setup();
  Tr = gsl_rng_default;
  r = gsl_rng_alloc (Tr);
    
  for (i = 0; i < N; i++) {
    double xi = gsl_rng_uniform(r) * 2. * M_PI;
    yi = gsl_spline_eval (spline, xi, acc);
    printf ("%g %g\n", xi, yi);
  }
    
  gsl_spline_free (spline);
  gsl_interp_accel_free (acc);
  gsl_rng_free (r);

  return 0;
}
