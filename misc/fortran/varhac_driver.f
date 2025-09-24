      PROGRAM RUNVARHAC
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER(I-N)
      INTEGER LDIM, KDIMMAX, KLDIM
      PARAMETER (LDIM=1000, KDIMMAX=10, KLDIM=10)
      REAL*8 DAT(LDIM,KDIMMAX), AAA(KLDIM,KDIMMAX)
      CHARACTER*256 DATAFILE, ARG
      INTEGER IMODEL, IMAX, IMEAN, IT1, IT2, NT, KDIM
      INTEGER I, J
      INTEGER IARGC
      IF (IARGC().LT.4) THEN
         WRITE(*,*) 'Usage: run_varhac datafile IMODEL IMAX IMEAN'
         STOP
      ENDIF
      CALL GETARG(1, DATAFILE)
      CALL GETARG(2, ARG)
      READ(ARG,*) IMODEL
      CALL GETARG(3, ARG)
      READ(ARG,*) IMAX
      CALL GETARG(4, ARG)
      READ(ARG,*) IMEAN
      OPEN(UNIT=10, FILE=DATAFILE, STATUS='OLD', FORM='FORMATTED')
      READ(10,*) NT, KDIM
      IF (NT.GT.LDIM) THEN
         WRITE(*,*) 'NT exceeds LDIM', NT, LDIM
         STOP
      ENDIF
      IF (KDIM.GT.KDIMMAX) THEN
         WRITE(*,*) 'KDIM exceeds KDIMMAX', KDIM, KDIMMAX
         STOP
      ENDIF
      DO I = 1, NT
         READ(10,*) (DAT(I,J), J=1,KDIM)
      END DO
      CLOSE(10)
      IT1 = 1
      IT2 = NT
      CALL VARHAC(DAT, LDIM, IT1, IT2, KDIM, IMODEL, IMAX,
     &     AAA, KLDIM, IMEAN)
      WRITE(*,'(A)') '--- VARHAC run ---'
      WRITE(*,'(A,A)') 'Data file: ', DATAFILE
      WRITE(*,'(A,I6)') 'NT: ', NT
      WRITE(*,'(A,I6)') 'KDIM: ', KDIM
      WRITE(*,'(A,I6)') 'IMODEL: ', IMODEL
      WRITE(*,'(A,I6)') 'IMAX: ', IMAX
      WRITE(*,'(A,I6)') 'IMEAN: ', IMEAN
      WRITE(*,'(A)') 'AAA matrix:'
      DO I = 1, KDIM
         WRITE(*,'(5(1X,ES15.6))') (AAA(I,J), J=1,KDIM)
      END DO
      END

C########################################################################
C
C
C       VARHAC.F, SUBROUTINE TO CALCULATE HAC VARIANCE-COVARIANCE MATRIX
C       (SPECTRAL DENSITY AT FREQUENCY ZERO) FOR A NUMBER OF SERIES
C
C		PROGRAM ASSUMES THAT A MEAN IS ZERO 
C
C		MAY 17, 1996
C 
C########################################################################


      SUBROUTINE VARHAC(DAT,LDIM,IT1,IT2,KDIM,IMODEL,IMAX,AAA,KLDIM
     &  ,IMEAN)
      IMPLICIT REAL*8(A-H,O-Z),INTEGER(I-N)
      PARAMETER(KMAX=10)
      PARAMETER(KMAX2=100)
      DIMENSION DAT(LDIM,KDIM),AAA(KLDIM,KDIM),
     &  BBB(KMAX,KMAX),ATEMP(KMAX,KMAX),
     &  CCC(KMAX,KMAX),BBBT(KMAX,KMAX),
     &  MINOR(KMAX),AIC(KMAX),XPX(KMAX2,KMAX2),ZAVE(KMAX)
      COMMON/SERIES/ ERROR(5000,KMAX),ERROR2(5000,KMAX)
      COMMON/ETA/ ETA(KMAX,KMAX,KMAX2)

C
C       DAT:            MATRIX FILLED IN THE MAIN PROGRAM WITH
C                       KDIM COLUMNS, THE DIMENSION STATEMENT IN
C                       THE MAIN PROGRAM MUST SAY THAT "DAT" HAS
C                       LDIM ROWS.
C       IT1:            FIRST OBSERVATION (ROW) OF DAT TO BE USED
C       IT2:            LAST OBSERVATION (ROW)  OF DAT TO BE USED
C       KDIM:           NUMBER OF COLUMNS OF DAT TO BE USED
C       IMODEL:         IF 1 THEN AIC IS USED
C                       IF 2 THEN BIC IS USED
C                       IF 3 THEN FIXED LAG-ORDER EQUAL TO IMAX IS USED
C       IMAX:           MAXIMUM LAG ORDER CONSIDERED
C       AAA:            COVARIANCE MATRIX (OUTPUT).  IN THE MAIN PROGRAM
C                       THE DIMENSION STATEMENT MUST SAY THAT "AAA"
C                       HAS KLDIM ROWS
C	 	IMEAN:		IF 1 THEN THE MEAN WILL BE SUBTRACTED
C

C
C       UNLIKE THE RATS PROGRAMS, THIS SUBROUTINE IS A LITTLE BIT MORE
C       EFFICIENT SINCE AFTER THE MODEL SELECTION IT REESTIMATES THE
C       AR FOR THE LONGEST POSSIBLE DATA SET.  WITH VARHAC2.F YOU GET
C       EXACTLY THE SAME OUTCOMES AS WITH THE RATS PROGRAMS
C

      IF(KDIM.GT.KMAX) THEN
      WRITE(*,*) 'KDIM TOO BIG, ADJUST KMAX '
      STOP
      ENDIF
      IF(IT2-IT1+1.GT.5000) THEN
      WRITE(*,*) 'NUMBER OF OBSERVATIONS TOO BIG, ADJUST DIMENSION'
      STOP
      ENDIF
      IF(KDIM*IMAX+1.GT.KMAX2) THEN
      WRITE(*,*) 'KDIM*IMAX TOO BIG, ADJUST KMAX '
      STOP
      ENDIF

      
C
C
C       FILL THE ERROR MATRIX
C
C

      NT = IT2 - IT1 + 1


      IF(IMEAN.EQ.1) THEN

      DO 31 K = 1,KDIM
      ZAVE(K) = 0.
      DO 32 I = 1,NT
      ZAVE(K) = ZAVE(K) + DAT(I+IT1-1,K)
32    CONTINUE
      ZAVE(K) = ZAVE(K)/DBLE(NT)
31    CONTINUE
      DO 101 I   =  1,NT
      DO 101 K   =  1,KDIM
      ERROR(I,K) =  DAT(I+IT1-1,K)-ZAVE(K)
101   CONTINUE
      ELSE
      DO 100 I   =  1,NT
      DO 100 K   =  1,KDIM
      ERROR(I,K) =  DAT(I+IT1-1,K)
100   CONTINUE
      ENDIF

      DO 10 J = 1,KDIM
      DO 10 K = 1,KDIM
      BBB(J,K) = 0.
      IF(J.EQ.K) BBB(J,K) = 1.
10    CONTINUE


      

C
C
C               MODEL SELECTION
C
C

      DO 3000 L = 1,KDIM

      IF(IMODEL.EQ.3) MINOR(L) = IMAX
      IF(IMODEL.EQ.3) GOTO 448

      AIC(L) = 0.
      DO 115 I = IMAX+1,NT
      AIC(L) = AIC(L) + ERROR(I,L)*ERROR(I,L)
115   CONTINUE
      AIC(L) = AIC(L)/DBLE(NT-IMAX)

      AMIN      = LOG(AIC(L)) 
      MINOR(L)  = 0

      DO 200 IORDER = 1,IMAX

      AIC(L) = 0.

      IPAR  = IORDER*KDIM 
      IFST  = IMAX+1
      ILAST = NT

      CALL AR(XPX,IFST,ILAST,IPAR,L,KDIM)

      DO 1115 I = IMAX+1,NT
      AIC(L) = AIC(L) + ERROR2(I,L)*ERROR2(I,L)
1115  CONTINUE

      AIC(L) = AIC(L)/DBLE(NT-IMAX)
      
      IF(IMODEL.EQ.1) THEN
      AIC(L) = LOG(AIC(L)) 
     $                 + 2.*DBLE(KDIM)*DBLE(IORDER)/DBLE(NT-IMAX)
      ELSE
      AIC(L) = LOG(AIC(L)) + LOG(DBLE(NT-IMAX))
     $                 *DBLE(KDIM)*DBLE(IORDER)/DBLE(NT-IMAX)
      ENDIF
  

      IF(AIC(L).LT.AMIN) THEN
      AMIN = AIC(L)
      MINOR(L) = IORDER
      ENDIF

200   CONTINUE

      
448   CONTINUE

c
c
c        REDO VAR FOR THE BEST MODEL AND LONGEST POSSIBLE DATA SET
c
c

      IFST  = MINOR(L) + 1
      IPAR  = MINOR(L)*KDIM 
      ILAST = NT

      IF(MINOR(L).GT.0) THEN
      CALL AR(XPX,IFST,ILAST,IPAR,L,KDIM)
      ELSE
      DO 210 I = 1,NT
      ERROR2(I,L) = ERROR(I,L)
210   CONTINUE
      ENDIF


3000  continue


      MIN = 0.

      DO 230 L = 1,KDIM
      IF(MINOR(L).GT.MIN) MIN = MINOR(L)
230   CONTINUE


      IFST = MIN + 1

      DO 2110 J = 1,KDIM
      DO 2110 K = 1,KDIM
      ATEMP(J,K) = 0.
2110  CONTINUE

      DO 2120 J = 1,KDIM
      DO 2120 K = J,KDIM
      DO 2115 I = IFST,NT
      ATEMP(J,K) = ATEMP(J,K) + ERROR2(I,J)*ERROR2(I,K)
2115  CONTINUE

      ATEMP(J,K) = ATEMP(J,K)/DBLE(NT-IFST+1)

2120  CONTINUE


      DO 2130 J = 2,KDIM
      DO 2130 K = 1,J-1
      ATEMP(J,K) = ATEMP(K,J)
2130  CONTINUE


C
C
C       GET VARIANCE-COVARIANCE ESTIMATOR
C
C

      IF(MIN.EQ.0) GOTO 334


      DO 300 L = 1,KDIM
      IF(MINOR(L).GT.0) THEN
      DO 301 J = 1,KDIM
      DO 301 I = 1,MINOR(L)
      BBB(L,J) =  BBB(L,J) - ETA(L,J,I)
301   CONTINUE
      ENDIF
300   CONTINUE



      CALL TRANSSQ(BBB,BBBT,KDIM,KMAX)

      CALL INVERTSQ(BBBT,KDIM,DET,KMAX)
      CALL INVERTSQ(BBB,KDIM,DET,KMAX)
      CALL MULTSQ(BBB,ATEMP,CCC,KDIM,KMAX)
      CALL MULTSQ(CCC,BBBT,ATEMP,KDIM,KMAX)

334   CONTINUE

      do 3333 j = 1,5         
      write(*,'(5f16.4)') (atemp(j,k),k=1,5)
3333  continue

      do 335 i = 1,kdim
      do 335 j = 1,kdim
      aaa(i,j) = atemp(i,j)
335   continue
      RETURN
      END


      SUBROUTINE AR(XPX,IFST,ILAST,IPAR,L,KDIM)
      IMPLICIT REAL*8(A-H,O-Z),INTEGER(I-N)
      PARAMETER(KMAX=10)
      PARAMETER(KMAX2=100)
      COMMON/SERIES/ ERROR(5000,KMAX),ERROR2(5000,KMAX)
      COMMON/ETA/ ETA(KMAX,KMAX,KMAX2)
      DIMENSION XPY(KMAX2),AX(5000,KMAX2),AY(5000),XPX(IPAR,IPAR),
     &  BETA(KMAX,KMAX2)

      IORDER = IPAR/KDIM


      DO 10 J = 1,IPAR
      XPY(J) = 0.0
      DO 10 K = 1,IPAR
      XPX(J,K) = 0.0
10    CONTINUE
      

      DO 20 I = IFST,ILAST
      AY(I)  =  ERROR(I,L)
      DO 20 J = 1,IORDER
      DO 20 K = 1,KDIM
      JJ = (J-1)*KDIM + K
      AX(I,JJ) =  ERROR(I-J,K)
20    CONTINUE

      DO 30 J = 1,IPAR
      DO 30 K = 1,IPAR
      DO 30 I = IFST,ILAST
      XPX(J,K) = XPX(J,K) + AX(I,J)*AX(I,K)
30    CONTINUE

      CALL INVERTSQ(XPX,IPAR,DET,IPAR)
      
      DO 40 J = 1,IPAR
      DO 40 I = IFST,ILAST
      XPY(J) = XPY(J) + AX(I,J)*AY(I)
40    CONTINUE

      DO 45 J = 1,IPAR
      BETA(L,J) = 0.0
      DO 45 K = 1,IPAR
      BETA(L,J) = BETA(L,J) + XPX(J,K)*XPY(K)
45    continue


      DO 50 I = IFST,ILAST
      ERROR2(I,L) = ERROR(I,L)
      DO 50 J = 1,IPAR
      ERROR2(I,L) = ERROR2(I,L) - AX(I,J)*BETA(L,J)
50    CONTINUE


      DO 60 J = 1,KDIM
      DO 60 I = 1,IORDER
      JJ = J + (I-1)*KDIM
      ETA(L,J,I) = BETA(L,JJ)
60    CONTINUE
      
      END



      SUBROUTINE TRANSSQ(E,F,NR,NMAX)

      IMPLICIT REAL*8(A-H,O-Z),INTEGER(I-N)
      DIMENSION E(NMAX,NMAX),F(NMAX,NMAX) 
      DO 800 J = 1,NR
      DO 800 K = 1,NR
      F(J,K) = E(K,J)
800   CONTINUE
      END


      SUBROUTINE MULTSQ(E , F , G , NR , NMAX )
      IMPLICIT REAL*8(A-H,O-Z)
      REAL*8 E(NMAX,NMAX) , F(NMAX,NMAX) , G(NMAX,NMAX)
      
      DO 831 J = 1,NR
      DO 832 I = 1,NR
      G(I,J) = 0.0
832   CONTINUE
831   CONTINUE
        
      DO 833 J = 1,NR
      DO 834 I = 1,NR
      DO 835 K = 1,NR
      G(I,J) = E(I,K)*F(K,J) + G(I,J)
835   CONTINUE
834   CONTINUE
833   CONTINUE
      RETURN
      END


      SUBROUTINE INVERTSQ(A,N,D,NMAX)

C
C
C     THIS SUBROUTINE COMPUTES THE INVERSE OF A MATRIX.
C
C
      IMPLICIT REAL*8(A-H,O-Z)
      DIMENSION A(NMAX,NMAX) ,L(100) , M(100)
      COMMON // ISING
      D=1.D0
      DO 80 K=1,N
      L(K)=K
      M(K)=K
      BIGA=A(K,K)
      DO 20 I=K,N
      DO 20 J=K,N
      IF(DABS(BIGA)-DABS(A(I,J))) 10,20,20
   10 BIGA=A(I,J)
      L(K)=I
      M(K)=J
   20 CONTINUE
      IF (DABS(BIGA).LT.(1.0E-15)) GO TO 99
      J=L(K)
      IF(L(K)-K) 35,35,25
   25 DO 30 I=1,N
      HOLD=-A(K,I)
      A(K,I)=A(J,I)
   30 A(J,I)=HOLD
   35 I=M(K)
      IF(M(K)-K) 45,45,37
   37 DO 40 J=1,N
      HOLD=-A(J,K)
      A(J,K)=A(J,I)
   40 A(J,I)=HOLD
   45 DO 55 I=1,N
      IF(I-K) 50,55,50
   50 A(I,K)=A(I,K)/(-A(K,K))
   55 CONTINUE
      DO 65 I=1,N
      DO 65 J=1,N
      IF(I-K) 57,65,57
   57 IF(J-K) 60,65,60
   60 A(I,J)=A(I,K)*A(K,J)+A(I,J)
   65 CONTINUE
      DO 75 J=1,N
      IF(J-K) 70,75,70
   70 A(K,J)=A(K,J)/A(K,K)
   75 CONTINUE
      D=D*A(K,K)
      A(K,K)=1.D0/A(K,K)
   80 CONTINUE
      K=N
  100 K=K-1
      IF(K) 150,150,103
  103 I=L(K)
      IF(I-K) 120,120,105
  105 DO 110 J=1,N
      HOLD=A(J,K)
      A(J,K)=-A(J,I)
  110 A(J,I)=HOLD
  120 J=M(K)
      IF(J-K) 100,100,125
  125 DO 130 I=1,N
      HOLD=A(K,I)
      A(K,I)=-A(J,I)
  130 A(J,I)=HOLD
      GO TO 100
   99 CONTINUE
      ISING = 1
      WRITE(11,991) ISING
 991  FORMAT('  SINGULAR MATRIX = = > ISING = ',I4)
  150 CONTINUE
      RETURN
      END


