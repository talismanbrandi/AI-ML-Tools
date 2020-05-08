#include <stdio.h>
#include <stdlib.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>

#include <gsl/gsl_rng.h>

int
main()
{
  const gsl_interp2d_type *T = gsl_interp2d_bilinear;
  const size_t N = 1000;      /* number of points to interpolate */
  const size_t nx = 1001;      /* x grid points */
  const size_t ny = 1001;      /* y grid points */
  double xa[nx];              /* define unit square */
  double ya[ny];
  double *za = malloc(nx * ny * sizeof(double));
  size_t i, j;
  gsl_spline2d *spline = gsl_spline2d_alloc(T, nx, ny);
  gsl_interp_accel *xacc = gsl_interp_accel_alloc();
  gsl_interp_accel *yacc = gsl_interp_accel_alloc();
    
  for (i = 0; i < nx; i++) {
    xa[i] = i * 2. * M_PI/(nx - 1.);
    for (j = 0; j < ny; j++) {
      ya[j] = j * 2. * M_PI/(ny - 1.);
      /* set z grid values */
      za[j*nx + i] = sin(5.*xa[i])*cos(6.*ya[j]);
    }
  }

  /* initialize interpolation */
  gsl_spline2d_init(spline, xa, ya, za, nx, ny);
    
  /* random number generator for generating output */
  const gsl_rng_type * Tr;
  gsl_rng * r;
  gsl_rng_env_setup();
  Tr = gsl_rng_default;
  r = gsl_rng_alloc (Tr);

  /* interpolate N values in x and y and print out grid for plotting */
  for (i = 0; i < N*N; i++) {
      double xi = gsl_rng_uniform(r) * 2. * M_PI;
      double yj = gsl_rng_uniform(r) * 2. * M_PI;
      double zij = gsl_spline2d_eval(spline, xi, yj, xacc, yacc);
//      printf("%f %f %f\n", xi, yj, zij);
//      printf("%f %f %f\n", xi, yj, (zij - sin(5.*xi)*cos(6.*yj))/zij*100);
  }

  gsl_spline2d_free(spline);
  gsl_interp_accel_free(xacc);
  gsl_interp_accel_free(yacc);
  free(za);
  gsl_rng_free (r);

  return 0;
}
