#ifndef DMIEV_H
#define DMIEV_H
#include <complex>
#include <assert.h>

typedef std::complex<double> dcmplx;

#define NMOMLEN 20201		/* =2*MAXTRM+1, PMON is of size [4][NMOMLEN] */

/*
     You need special compiler options to promote real to double precision etc !!!

      SUBROUTINE MIEV0( XX, CREFIN, PERFCT, MIMCUT, ANYANG, NUMANG, XMU,
     &                  NMOM, IPOLZN, MOMDIM, PRNT, QEXT, QSCA, GQSC,
     &                  PMOM, SFORW, SBACK, S1, S2, TFORW, TBACK,
     &                  SPIKE )
      LOGICAL  ANYANG, PERFCT, PRNT(*)
      INTEGER  IPOLZN, MOMDIM, NUMANG, NMOM
      REAL     GQSC, MIMCUT, PMOM( 0:MOMDIM, * ), QEXT, QSCA, SPIKE,
     &         XMU(*), XX
      COMPLEX  CREFIN, SFORW, SBACK, S1(*), S2(*), TFORW(*), TBACK(*)
*/
extern "C" {
    void miev0_( double* XX, dcmplx* CREFIN, int* PERFCT, double* MIMCUT, int* ANYANG, int* NUMANG, double* XMU, int* NMOM, int* IPOLZN, int* MOMDIM, int* PRNT, double* QEXT, double* QSCA, double* GQSC, double* PMOM, dcmplx* SFORW, dcmplx* SBACK, dcmplx* S1, dcmplx* S2, dcmplx* TFORW, dcmplx* TBACK, double* SPIKE );
    void amiev(double* xx, dcmplx* crefin, double* mu, dcmplx* s1, dcmplx* s2);
    void mievp(double xx, dcmplx crefin, int numang, dcmplx* s1, dcmplx* s2, double pmom[4][NMOMLEN]);
    void miev(double xx, dcmplx crefin, int numang, double* mu, dcmplx* s1, dcmplx* s2);
    void mievinfo(double xx, dcmplx crefin, double* qext, double* qsca, double* g);
};

#endif
