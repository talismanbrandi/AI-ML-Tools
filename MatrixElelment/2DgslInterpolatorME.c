#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_rng.h>

int
main()
{
    const gsl_interp2d_type *T = gsl_interp2d_bilinear;
    const size_t nx = 1001;      /* x grid points */
    const size_t ny = 1001;      /* y grid points */
    double xa[nx];              /* define unit square */
    double ya[ny];
    double *za = malloc(nx * ny * sizeof(double));
    double tmpa, tmpb, tmpc;
    bool print = false;
    double accuracy = 0;
    size_t i, j;
    gsl_spline2d *spline = gsl_spline2d_alloc(T, nx, ny);
    gsl_interp_accel *xacc = gsl_interp_accel_alloc();
    gsl_interp_accel *yacc = gsl_interp_accel_alloc();
    
    FILE *in_file  = fopen("data/ggzz_grid_gsl.dat", "r"); // read only
    if (in_file == NULL) {
        printf("Error! Could not open file\n");
        exit(-1); // must include stdlib.h
    }
    
    
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            fscanf(in_file, "%lf %lf %lf", &xa[i], &ya[j], &za[j*nx + i]);
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

    for (i = 0; i < nx-1; i++) {
      for (j = 0; j < ny-1; j++) {
          double xi = (xa[i]+xa[i+1])/2.;
          double yj = (ya[j]+ya[j+1])/2.;
          double zij = gsl_spline2d_eval(spline, xi, yj, xacc, yacc);
          double z_m = (za[j*nx + i]+za[(j+1)*nx + (i+1)])/2.;
          accuracy += fabs(zij-z_m/z_m);
          if (print) printf("%lf %lf %lf\n", xi, yj, (zij-z_m)/z_m*100);
      }
    }
    accuracy /= (nx-1)*(ny-1);
    printf("%lf \n", (accuracy)*100.);

    gsl_spline2d_free(spline);
    gsl_interp_accel_free(xacc);
    gsl_interp_accel_free(yacc);
    free(za);
    gsl_rng_free (r);

    return 0;
}
