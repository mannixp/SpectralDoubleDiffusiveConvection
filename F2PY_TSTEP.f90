! ~~~~ f2py -c --fcompiler=intelem --opt='-O3' -llapack F2PY_TSTEP.f90 -m LT_STEP ~~~~ Serial

! ~~~~ f2py -c --fcompiler=intelem --opt='-O3' --f90flags='-fast' -llapack F2PY_TSTEP.f90 -m LT_STEP ~~~~ Serial

!~~~~GNU95 ~~~~~~~~~ has problem with fractions
! ~~~~ f2py -c --fcompiler=gnu95 --opt='-O3' --f90flags='-fopenmp' -lgomp F2PY_TSTEP.f90 -m LT_STEP~~~~ To compile


! For laptop
!f2py -c --fcompiler=intelem --opt='-Ofast' --f90flags='-fast -mkl=parallel -ipo' -liomp5 -lpthread -lm -ldl F2PY_TSTEP.f90 -m LT_STEP


module N_LIN
 !use types, only : dp, pi
 !use utils, only : stop_error
 !use omp_lib
 !use iso_c_binding
 !!!makes OpenMP routines, variables available
 implicit none
 !integer :: i1,j1, k1
 !integer :: NumThreads,threadID
 save
!private
contains

!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~

subroutine T0_JT(IR,nr,N_fm,g,f)
 
 integer, intent(in) :: N_fm,nr
 real(kind=8), dimension(:), intent(in) :: IR
 real(kind=8), dimension(:), intent(in) :: g
 real(kind=8), dimension(nr*N_fm), intent(out) :: f
 
 real(kind=8), dimension(nr) :: b
 integer :: jj,ind_j,ind_k,j
 f = 0.d0; 

 !~~~~~~~~~~~~~~~~~~#
 b = 0.d0; !Do ODDS
 do jj = 0,(N_fm-1),2

  j = (N_fm - (jj + 1) );
  ind_j = j*nr;
  ind_k = (j -1)*nr;
		
  IF (j < (N_fm - 1 ) ) THEN
    b = b + g( 1+ ind_k+2*nr:ind_k+3*nr);	
  END IF
		
  IF (j == 0) THEN
    f(1+ind_j:ind_j+nr) = IR*b;
  ELSE
    f(1+ind_j:ind_j+nr) = IR*( float(j + 1)*g(1+ind_k:ind_k+nr) + 2.d0*b );	
  END IF

 end do
 
 !~~~~~~~~~~~~~~~~~~#
 b = 0.d0; !Do ODDS
 do jj = 1,(N_fm-1),2

  j = (N_fm - (jj + 1) );
  ind_j = j*nr;
  ind_k = (j -1)*nr;
		
  IF (j < (N_fm - 1 ) ) THEN
    b = b + g( 1+ ind_k+2*nr:ind_k+3*nr);	
  END IF
		
  IF (j == 0) THEN
    f(1+ind_j:ind_j+nr) = IR*b;
  ELSE
    f(1+ind_j:ind_j+nr) = IR*( float(j + 1)*g(1+ind_k:ind_k+nr) + 2.d0*b );	
  END IF

 end do
	
end subroutine T0_JT


!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~

subroutine A2_SINE(D2,IR2,nr,N_fm,g,f)
 
 integer, intent(in) :: N_fm,nr
 real(kind=8), dimension(:,:), intent(in) :: D2,IR2
 real(kind=8), dimension(:), intent(in) :: g
 real(kind=8), dimension(nr*N_fm), intent(out) :: f
 
 real(kind=8), dimension(nr) :: f_e
 integer :: jj,ind_j,j
 f = 0.d0; !jj = 0; row = 0; ind_j = 0;

 !~~~~~~~~~~~~~~~~~~#
 f_e = 0.d0; !Do ODDS
 do jj = 0,(N_fm-1),2

  j = N_fm-jj; 
  ind_j = (j-1)*nr; !# Row ind
							
  f(1+ ind_j:ind_j+nr) = matmul(D2,g(1+ind_j:ind_j+nr)) -float(j)*matmul(IR2,float((j+1))*g(1+ind_j:ind_j+nr)+ 2.d0*f_e);
  f_e = f_e + g(1+ind_j:ind_j+nr);
  
 end do
 
 !~~~~~~~~~~~~~~~~~~#
 f_e = 0.d0; !Do_EVENS
 do jj = 1,(N_fm-1),2

  j = N_fm-jj; 
  ind_j = (j-1)*nr; !# Row ind
							
  f(1+ ind_j:ind_j+nr) = matmul(D2,g(1+ind_j:ind_j+nr)) -float(j)*matmul(IR2,float((j+1))*g(1+ind_j:ind_j+nr)+ 2.d0*f_e);
  f_e = f_e + g(1+ind_j:ind_j+nr);
  
 end do
	
end subroutine A2_SINE

!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~

! ***** Solves -Nabla^2*f = g
subroutine NAB2_TSTEP(dt,N_fm,nr,N2_INV,g,f)

 integer, intent(in) :: N_fm,nr
 real(kind=8), intent(in) :: dt
 real(kind=8), dimension(:,:,:), intent(in) :: N2_INV
 real(kind=8), dimension(:), intent(in) :: g
 real(kind=8), dimension(nr*N_fm), intent(out) :: f
 
 real(kind=8), dimension(nr) :: b	
 integer :: jj,row,ind_j;
 f = 0.d0; !jj = 0; row = 0; ind_j = 0;
 
 
 b = 0.d0; ! Do Evens
 do jj = 0,(N_fm-1),2

  row = (N_fm - (jj + 1) );
  ind_j = row*nr;
    
  IF ( (row > 0) .AND. (row < (N_fm - 2 ) ) ) THEN
   b = b + (2.d0)*(float(row) + 2.d0)*f( 1+ (row+2)*nr:(row+3)*nr );
  ELSE IF (row == 0) THEN
   b = (0.5)*b + (float(row) + 2.d0)*f( 1 + (row+2)*nr:(row+3)*nr );
  END IF
 	
  f(1+ind_j:ind_j+nr) = matmul( N2_INV(:,:,1+jj) , g( 1+ ind_j:ind_j+nr)-dt*b );
 end do
 
 b = 0.d0; ! Do Odds
 do jj = 1,(N_fm-1),2

  row = (N_fm - (jj + 1) );
  ind_j = row*nr;
    
  IF ( (row > 0) .AND. (row < (N_fm - 2 ) ) ) THEN
   b = b + (2.d0)*(float(row) + 2.d0)*f( 1+ (row+2)*nr:(row+3)*nr );
  ELSE IF (row == 0) THEN
   b = (0.5)*b + (float(row) + 2.d0)*f( 1 + (row+2)*nr:(row+3)*nr );
  END IF

  f(1+ind_j:ind_j+nr) = matmul( N2_INV(:,:,1+jj) , g( 1+ ind_j:ind_j+nr)-dt*b );
 end do
 
end subroutine NAB2_TSTEP

!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~

! ***** Solves -Pr*A^4*f = g
subroutine A4_TSTEP(dt,N_fm,nr,A4_INV,D2,IR4,IR2,g,f)

 real(kind=8), intent(in) :: dt
 integer, intent(in) :: N_fm,nr
 real(kind=8), dimension(:,:,:), intent(in) :: A4_INV
 real(kind=8), dimension(:,:), intent(in) :: D2,IR4,IR2
 real(kind=8), dimension(:), intent(in) :: g
 real(kind=8), dimension(nr*N_fm), intent(out) :: f
 
 real(kind=8), dimension(nr) :: f_e,bf_e,b_test
 real(kind=8), dimension(nr,nr) :: L1;	
 real(kind=8) :: bj,bjt,j
 integer :: jj,row,ind_j;
 f = 0.d0; 
 
 !~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~##~~~~~~~~~~~~~~~~~~~~~
 L1 = 0.d0;
 f_e = 0.d0; bf_e = 0.d0; b_test = 0.d0; ! Do ODDS
 bj = 0.d0; bjt = 0.d0; j = 0.d0;

 do jj = 0,(N_fm-1),2

  row = (N_fm - (jj + 1) ); 
  ind_j = row*nr; !# Row ind
  j = float(N_fm-jj); ! Remeber sine counting is from 1 - N_theta
  bj = -j*(j + 1.d0); bjt = -2.d0*j;
  
  L1 = D2 + bj*IR4;
    
  IF (row < (N_fm - 2 ) ) THEN
   
   f_e = f_e + f( 1+ (row+2)*nr:(row+3)*nr);

   b_test = dt*bjt*( matmul(L1, f_e ) + matmul(IR4, bf_e) ) - bjt*matmul(IR2,f_e);
   
   !f(1+ ind_j:ind_j+nr) = Ax(-L,g(1+ind_j:ind_j+nr)+b_test,nr);   !#O(Nr^3 N_theta)

   f(1+ ind_j:ind_j+nr) = matmul(A4_INV(:,:,1+jj),g(1+ind_j:ind_j+nr)+b_test);

   !# Add sums after to get +2 lag
   bf_e = bf_e + bj*f( 1+ ind_j:ind_j+nr) + bjt*f_e;	
  ELSE
   !f(1+ ind_j:ind_j+nr) = Ax(-L,g(1+ind_j:ind_j+nr),nr);
   f(1+ ind_j:ind_j+nr) = matmul(A4_INV(:,:,1+jj),g(1+ind_j:ind_j+nr) );
   bf_e = bf_e + bj*f( 1+ ind_j:ind_j+nr);
  END IF

 end do 

 
 !~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~##~~~~~~~~~~~~~~~~~~~~~
 L1 = 0.d0;
 f_e = 0.d0; bf_e = 0.d0; b_test = 0.d0; ! Do Evens
 bj = 0.d0; bjt = 0.d0; j = 0.d0;

 do jj = 1,(N_fm-1),2

  row = (N_fm - (jj + 1) ); 
  ind_j = row*nr; !# Row ind
  j = float(N_fm-jj); ! Remeber sine counting is from 1 - N_theta
  bj = -j*(j + 1.d0); bjt = -2.d0*j;
  
  L1 = D2 + bj*IR4; 
     
  IF (row < (N_fm - 2 ) ) THEN
   
   f_e = f_e + f( 1+ (row+2)*nr:(row+3)*nr);

   b_test = dt*bjt*( matmul(L1, f_e ) + matmul(IR4, bf_e) ) - bjt*matmul(IR2,f_e);
   
   f(1+ ind_j:ind_j+nr) = matmul(A4_INV(:,:,1+jj),g(1+ind_j:ind_j+nr)+b_test);

   !# Add sums after to get +2 lag
   bf_e = bf_e + bj*f( 1+ ind_j:ind_j+nr) + bjt*f_e;	
  ELSE
  
   f(1+ ind_j:ind_j+nr) = matmul(A4_INV(:,:,1+jj),g(1+ind_j:ind_j+nr) );
   bf_e = bf_e + bj*f( 1+ ind_j:ind_j+nr);
  END IF

 end do 
 
end subroutine A4_TSTEP


!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~
! ***** Symmetric counterpatrts and initialize as global
!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~


!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~

subroutine T0_JT_SYM(IR,nr,N_fm,g,f)
 
 integer, intent(in) :: N_fm,nr
 real(kind=8), dimension(:), intent(in) :: IR
 real(kind=8), dimension(:), intent(in) :: g
 real(kind=8), dimension(nr*N_fm), intent(out) :: f
 
 real(kind=8), dimension(nr) :: b
 integer :: jj,ind_j,ind_k,j
 f = 0.d0; 

 !~~~~~~~~~~~~~~~~~~#
 b = 0.d0; !Do Evens
 do jj = 1,(N_fm-1),2

  j = (N_fm - (jj + 1) );
  ind_j = j*nr;
  ind_k = (j -1)*nr;
		
  IF (j < (N_fm - 1 ) ) THEN
    b = b + g( 1+ ind_k+2*nr:ind_k+3*nr);	
  END IF
		
  IF (j == 0) THEN
    f(1+ind_j:ind_j+nr) = IR*b;
  ELSE
    f(1+ind_j:ind_j+nr) = IR*( float(j + 1)*g(1+ind_k:ind_k+nr) + 2.d0*b );	
  END IF

 end do
	
end subroutine T0_JT_SYM


!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~

subroutine A2_SINE_SYM(D2,IR2,nr,N_fm,g,f)
 
 integer, intent(in) :: N_fm,nr
 real(kind=8), dimension(:,:), intent(in) :: D2,IR2
 real(kind=8), dimension(:), intent(in) :: g
 real(kind=8), dimension(nr*N_fm), intent(out) :: f
 
 real(kind=8), dimension(nr) :: f_e
 integer :: jj,ind_j,j
 f = 0.d0; !jj = 0; row = 0; ind_j = 0;

 !~~~~~~~~~~~~~~~~~~#
 f_e = 0.d0; !Do EVENS
 do jj = 0,(N_fm-1),2

  j = N_fm-jj; 
  ind_j = (j-1)*nr; !# Row ind
							
  f(1+ ind_j:ind_j+nr) = matmul(D2,g(1+ind_j:ind_j+nr)) -float(j)*matmul(IR2,float((j+1))*g(1+ind_j:ind_j+nr)+ 2.d0*f_e);
  f_e = f_e + g(1+ind_j:ind_j+nr);
  
 end do
	
end subroutine A2_SINE_SYM

!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~

! ***** Solves -Nabla^2*f = g
subroutine NAB2_TSTEP_SYM(dt,N_fm,nr,N2_INV,g,f)

 integer, intent(in) :: N_fm,nr
 real(kind=8), intent(in) :: dt
 real(kind=8), dimension(:,:,:), intent(in) :: N2_INV
 real(kind=8), dimension(:), intent(in) :: g
 real(kind=8), dimension(nr*N_fm), intent(out) :: f
 
 real(kind=8), dimension(nr) :: b	
 integer :: jj,row,ind_j;
 f = 0.d0; !jj = 0; row = 0; ind_j = 0;
 
 
 b = 0.d0; ! Do Evens
 do jj = 1,(N_fm-1),2

  row = (N_fm - (jj + 1) );
  ind_j = row*nr;
    
  IF ( (row > 0) .AND. (row < (N_fm - 2 ) ) ) THEN
   b = b + (2.d0)*(float(row) + 2.d0)*f( 1+ (row+2)*nr:(row+3)*nr );
  ELSE IF (row == 0) THEN
   b = (0.5)*b + (float(row) + 2.d0)*f( 1 + (row+2)*nr:(row+3)*nr );
  END IF

  f(1+ind_j:ind_j+nr) = matmul( N2_INV(:,:,1+jj) , g( 1+ ind_j:ind_j+nr)-dt*b );
 end do
 
end subroutine NAB2_TSTEP_SYM

!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~

! ***** Solves -Pr*A^4*f = g
subroutine A4_TSTEP_SYM(dt,N_fm,nr,A4_INV,D2,IR4,IR2,g,f)

 real(kind=8), intent(in) :: dt
 integer, intent(in) :: N_fm,nr
 real(kind=8), dimension(:,:,:), intent(in) :: A4_INV
 real(kind=8), dimension(:,:), intent(in) :: D2,IR4,IR2
 real(kind=8), dimension(:), intent(in) :: g
 real(kind=8), dimension(nr*N_fm), intent(out) :: f
 
 real(kind=8), dimension(nr) :: f_e,bf_e,b_test
 real(kind=8), dimension(nr,nr) :: L1;	
 real(kind=8) :: bj,bjt,j
 integer :: jj,row,ind_j;
 f = 0.d0; 
 
 !~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~##~~~~~~~~~~~~~~~~~~~~~
 L1 = 0.d0;
 f_e = 0.d0; bf_e = 0.d0; b_test = 0.d0; ! Do Evens
 bj = 0.d0; bjt = 0.d0; j = 0.d0;

 do jj = 0,(N_fm-1),2

  row = (N_fm - (jj + 1) ); 
  ind_j = row*nr; !# Row ind
  j = float(N_fm-jj); ! Remeber sine counting is from 1 - N_theta
  bj = -j*(j + 1.d0); bjt = -2.d0*j;
  
  L1 = D2 + bj*IR4;
    
  IF (row < (N_fm - 2 ) ) THEN
   
   f_e = f_e + f( 1+ (row+2)*nr:(row+3)*nr);

   b_test = dt*bjt*( matmul(L1, f_e ) + matmul(IR4, bf_e) ) - bjt*matmul(IR2,f_e);
   
   !f(1+ ind_j:ind_j+nr) = Ax(-L,g(1+ind_j:ind_j+nr)+b_test,nr);   !#O(Nr^3 N_theta)

   f(1+ ind_j:ind_j+nr) = matmul(A4_INV(:,:,1+jj),g(1+ind_j:ind_j+nr)+b_test);

   !# Add sums after to get +2 lag
   bf_e = bf_e + bj*f( 1+ ind_j:ind_j+nr) + bjt*f_e;	
  ELSE
   !f(1+ ind_j:ind_j+nr) = Ax(-L,g(1+ind_j:ind_j+nr),nr);
   f(1+ ind_j:ind_j+nr) = matmul(A4_INV(:,:,1+jj),g(1+ind_j:ind_j+nr) );
   bf_e = bf_e + bj*f( 1+ ind_j:ind_j+nr);
  END IF

 end do 

end subroutine A4_TSTEP_SYM



end module N_LIN
