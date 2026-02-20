#include <luminis/mie/dmiev.h>
#include <iostream>

extern "C" {
/* calls miev0 with one given mu=cos(theta) */
void amiev(double *xx, dcmplx *crefin, double *mu, dcmplx *s1, dcmplx *s2) {
  int PERFCT = 0;
  double MIMCUT = 1e-8;
  int ANYANG = 1;
  int NUMANG = 1;
  int NMOM = 0;
  int IPOLZN = 0;
  int MOMDIM = 1;
  int PRNT[2] = {0, 0};
  double QEXT, QSCA, GQSC, PMOM, SPIKE;
  dcmplx SFORW, SBACK, TFORW[2], TBACK[2];
  miev0_(xx, crefin, &PERFCT, &MIMCUT, &ANYANG, &NUMANG, mu, &NMOM, &IPOLZN,
         &MOMDIM, PRNT, &QEXT, &QSCA, &GQSC, &PMOM, &SFORW, &SBACK, s1, s2,
         TFORW, TBACK, &SPIKE);
};

/* calls miev0 with one given mu=cos(theta) and obtain Legendre coefficients of
 * S21 */
/* pmon is a pointer to array pmom[4][nmom]. numang must be odd */
void mievp(double xx, dcmplx crefin, int numang, dcmplx *s1, dcmplx *s2, double pmom[4][NMOMLEN]) {
  int PERFCT = 0;
  double MIMCUT = 1e-8;
  int ANYANG = 0;
  int IPOLZN = 3; /* calculates S21 */
  int MOMDIM = NMOMLEN - 1;
  int PRNT[2] = {0, 0};
  double QEXT, QSCA, GQSC, SPIKE;
  dcmplx SFORW, SBACK, TFORW[2], TBACK[2];
  int nmom = (int)floor(
      2. * (xx + 4. * pow(xx, 1. / 3.) +
            2.)); /* counting from 0 to nmom inclusively in Fortran */
  double step = 2. / (numang - 1);
  double *mu = new double[numang];
  for (int i = 0; i < numang; i++)
    mu[i] = 1. - i * step;
  assert(nmom > 2 * (xx + 4 * pow(xx, 1 / 3.) + 1));
  miev0_(&xx, &crefin, &PERFCT, &MIMCUT, &ANYANG, &numang, mu, &nmom, &IPOLZN,
         &MOMDIM, PRNT, &QEXT, &QSCA, &GQSC, &pmom[0][0], &SFORW, &SBACK, s1,
         s2, TFORW, TBACK, &SPIKE);
  delete[] mu;
};

/* calls miev0 with n given mu=cos(theta) */
void miev(double xx, dcmplx crefin, int numang, double *mu, dcmplx *s1, dcmplx *s2) {
  int PERFCT = 0;
  double MIMCUT = 1e-8;
  int ANYANG = 1;
  int NMOM = 0;
  int IPOLZN = 0;
  int MOMDIM = 1;
  int PRNT[2] = {0, 0};
  double QEXT, QSCA, GQSC, PMOM, SPIKE;
  dcmplx SFORW, SBACK, TFORW[2], TBACK[2];
  miev0_(&xx, &crefin, &PERFCT, &MIMCUT, &ANYANG, &numang, mu, &NMOM, &IPOLZN,
         &MOMDIM, PRNT, &QEXT, &QSCA, &GQSC, &PMOM, &SFORW, &SBACK, s1, s2,
         TFORW, TBACK, &SPIKE);
};

/* obtain the basic info */
void mievinfo(double xx, dcmplx crefin, double *qext, double *qsca, double *g) {
  int PERFCT = 0;
  double MIMCUT = 1e-8;
  int ANYANG = 1;
  int NUMANG = 1;
  double XMU = 0;
  int NMOM = 0;
  int IPOLZN = 0;
  int MOMDIM = 1;
  int PRNT[2] = {0, 0};
  double gqsc, PMOM, SPIKE;
  dcmplx SFORW, SBACK, S1, S2, TFORW[2], TBACK[2];
  miev0_(&xx, &crefin, &PERFCT, &MIMCUT, &ANYANG, &NUMANG, &XMU, &NMOM, &IPOLZN,
         &MOMDIM, PRNT, qext, qsca, &gqsc, &PMOM, &SFORW, &SBACK, &S1, &S2,
         TFORW, TBACK, &SPIKE);
  *g = gqsc / (*qsca);
};
};
