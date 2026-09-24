
      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
!     ---------- ----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)

      DOUBLE PRECISION S, tau, B, Fert, alpha, T, alpha_max, rho, S_T, g_F, D_t, D0, zeta, time, B_low, B_high, t_phase

      ! Define the state variable
      S = U(1)
      time = U(2)

      Fert = PAR(1)
      tau  = PAR(2)
      B_high = PAR(6)
      B_low = PAR(7)
      T = PAR(11)

      ! Piecewise-constant forcing over each unit interval [k, k+1):
      ! B = B_low on [k, k+tau], B = B_high on (k+tau, k+1).
      t_phase = MOD(time, 1.0d0)
      IF (t_phase .LE. tau) THEN
            B = B_low
      ELSE
            B = B_high
      END IF

      !write(*,*) B

      alpha_max = 0.25d0
      rho = 50.d0
      S_T = 0.2d0
      D0 = 1.2D0
      zeta = 1.0d0
      g_F = Fert**2 - 2*zeta*Fert + D0
      D_t = g_F * B

      IF (S >= 1) THEN
            ! To ensure S is always strictly less than 1
            S = 0.9999999
      END IF

      alpha = alpha_max / (1 + exp(-rho * (S - S_T)))
      
      F(1) = T * (alpha - D_t) * S * (1.0d0 - S)
      F(2) = T
  
      END SUBROUTINE FUNC

! ----------------------------------------------------------------------
! ----------------------------------------------------------------------

      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     ---------- -----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

      !              Fert     tau
      PAR(1:2) = (/ 1.25d0, 0.3d0 /)
      PAR(6) = 0.9d0  ! B_high
      PAR(7) = 0.1d0  ! B_low
      PAR(11) = 0.d0

       U(1) = 0.21d0
       U(2) = 0.0d0

       PAR(3) = U(1) ! S(0)
       PAR(4) = U(1) ! S(1)
       PAR(5) = 0.0d0
       PAR(8) = 1.0d0 ! Need a better estimate here

      END SUBROUTINE STPNT

! ----------------------------------------------------------------------
! ----------------------------------------------------------------------
      SUBROUTINE BCND(NDIM,PAR,ICP,NBC,U0,U1,FB,IJAC,DBC) 
!     ---------- ---- 

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM, ICP(*), NBC, IJAC
      DOUBLE PRECISION, INTENT(IN) :: PAR(*), U0(NDIM), U1(NDIM)
      DOUBLE PRECISION, INTENT(OUT) :: FB(NBC)
      DOUBLE PRECISION, INTENT(INOUT) :: DBC(NBC,*)
       FB(1) = U0(1) - PAR(3) 
       FB(2) = U1(1) - PAR(4) ! This is just to monitor S(T)
       FB(3) = U0(2) ! t(0) = 0
       FB(4) = PAR(4) - PAR(3) - PAR(5)
       ! For some reason this does not work! 
       ! FB(4) = U1(1) - U0(1) - PAR(5)

      END SUBROUTINE BCND

! ----------------------------------------------------------------------
! ----------------------------------------------------------------------
      SUBROUTINE PVLS(NDIM,U,PAR)
!     ---------- ----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: PAR(*)

      DOUBLE PRECISION, EXTERNAL :: GETP

      END SUBROUTINE PVLS
! ----------------------------------------------------------------------
! ----------------------------------------------------------------------
! ----------------------------------------------------------------------
! ----------------------------------------------------------------------
      SUBROUTINE ICND(NDIM,PAR,ICP,NINT,U,UOLD,UDOT,UPOLD,FI,IJAC,DINT) 
!     ---------- ---- 

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM, ICP(*), NINT, IJAC
      DOUBLE PRECISION, INTENT(IN) :: PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), UOLD(NDIM), UDOT(NDIM), UPOLD(NDIM)
      DOUBLE PRECISION, INTENT(OUT) :: FI(NINT)
      DOUBLE PRECISION, INTENT(INOUT) :: DINT(NINT,*)
      DOUBLE PRECISION S, tau, B, Fert, alpha, T, alpha_max, rho, S_T, g_F, D_t, D0, zeta, time, B_low, B_high, t_phase, RHS

      ! Define the state variable
      S = U(1)
      time = U(2)

      Fert = PAR(1)
      tau  = PAR(2)
      B_high = PAR(6)
      B_low = PAR(7)
      T = PAR(11)

      ! Piecewise-constant forcing over each unit interval [k, k+1):
      ! B = B_low on [k, k+tau], B = B_high on (k+tau, k+1).
      t_phase = MOD(time, 1.0d0)
      IF (t_phase .LE. tau) THEN
            B = B_low
      ELSE
            B = B_high
      END IF

      alpha_max = 0.25d0
      rho = 50.d0
      S_T = 0.2d0
      D0 = 1.2D0
      zeta = 1.0d0
      g_F = Fert**2 - 2*zeta*Fert + D0
      D_t = g_F * B

      IF (S >= 1) THEN
            ! To ensure S is always strictly less than 1
            S = 0.9999999
      END IF

      alpha = alpha_max / (1 + exp(-rho * (S - S_T)))
      
      RHS = T * (alpha - D_t) * S * (1.0d0 - S)

      FI(1)= RHS - PAR(8)

      END SUBROUTINE ICND
! ----------------------------------------------------------------------
! ----------------------------------------------------------------------
      SUBROUTINE FOPT
      END SUBROUTINE FOPT
