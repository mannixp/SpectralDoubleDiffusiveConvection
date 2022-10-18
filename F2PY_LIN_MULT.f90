!~~~~ IFORT ~~~~~~~~

! USE https://software.intel.com/content/www/us/en/develop/articles/intel-mkl-link-line-advisor.html
! to obtain compile flags and linker options

! Additional flags are -ipo and mutune

! For laptop
!f2py -c --fcompiler=intelem --opt='-Ofast' --f90flags='-fast -mkl=parallel -ipo' -liomp5 -lpthread -lm -ldl F2PY_LIN_MULT.f90 -m L_PARA

! For cluster
! ~~~~ f2py -c --fcompiler=intelem --opt='-Ofast' F2PY_LIN_MULT.f90 -m L_PARA ~~~~ To compile Parrallel, can change openmp for -parallel

!adding --f90flags='-fast' compile flag works well
! ~~~~ f2py -c --fcompiler=intelem --opt='-O3' --f90flags='-fast' F2PY_LIN_MULT.f90 -m L_PARA ~~~~ Serial

!~~~~GNU95 ~~~~~~~~~ has problem with fractions
! ~~~~ f2py -c --fcompiler=gnu95 --opt='-O3' F2PY_LIN_MULT.f90 -m L_PARA ~~~~ To compile


module NON_LIN1
 !use types, only : dp, pi
 !use utils, only : stop_error
 !!use omp_lib
 !!use iso_c_binding
 !!!makes OpenMP routines, variables available
 implicit none
 integer :: i1,j1, k1
 !integer :: NumThreads,threadID
 save
!private
contains

!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~

subroutine A2_SINE(D2,IR4,nr,N_fm,g,f)
 implicit none
 integer, intent(in) :: N_fm,nr
 real(kind=8), dimension(:,:), intent(in) :: D2,IR4
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
							
  f(1+ ind_j:ind_j+nr) = matmul(D2,g(1+ind_j:ind_j+nr)) -float(j)*matmul(IR4,float((j+1))*g(1+ind_j:ind_j+nr)+ 2.d0*f_e);
  f_e = f_e + g(1+ind_j:ind_j+nr);
  
 end do
 
 !~~~~~~~~~~~~~~~~~~#
 f_e = 0.d0; !Do_EVENS
 do jj = 1,(N_fm-1),2

  j = N_fm-jj; 
  ind_j = (j-1)*nr; !# Row ind
							
  f(1+ ind_j:ind_j+nr) = matmul(D2,g(1+ind_j:ind_j+nr)) -float(j)*matmul(IR4,float((j+1))*g(1+ind_j:ind_j+nr)+ 2.d0*f_e);
  f_e = f_e + g(1+ind_j:ind_j+nr);
  
 end do
	
end subroutine A2_SINE



subroutine JT(nr,N_fm,g,f)
 implicit none
 integer, intent(in) :: N_fm,nr
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
    f(1+ind_j:ind_j+nr) = b;
  ELSE
    f(1+ind_j:ind_j+nr) = float(j + 1)*g(1+ind_k:ind_k+nr) + 2.d0*b;	
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
    f(1+ind_j:ind_j+nr) = b;
  ELSE
    f(1+ind_j:ind_j+nr) = float(j + 1)*g(1+ind_k:ind_k+nr) + 2.d0*b;	
  END IF

 end do
	
end subroutine JT



subroutine T0_JT(IR,nr,N_fm,g,f)
 implicit none
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


! ***** Solves -Nabla^2*f = g
subroutine nab2_V2(N_fm,nr,N2_INV,g,f)
 implicit none
 integer, intent(in) :: N_fm,nr
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
 
  !f(1+ind_j:ind_j+nr) = Ax( -(N2R -float(row)*( float(row) + 1.d0)*I ),g( 1+ ind_j:ind_j+nr)-b  ,nr);	
  f(1+ind_j:ind_j+nr) = matmul( N2_INV(:,:,1+jj) , g( 1+ ind_j:ind_j+nr)-b );
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

  !f(1+ind_j:ind_j+nr) = Ax( -(N2R -float(row)*( float(row) + 1.d0)*I ),g( 1+ ind_j:ind_j+nr)-b  ,nr);	
  f(1+ind_j:ind_j+nr) = matmul( N2_INV(:,:,1+jj) , g( 1+ ind_j:ind_j+nr)-b );
 end do
 
end subroutine nab2_V2

! ***** Solves -Pr*A^4*f = g
subroutine A4_BSub_V2(N_fm,nr,A4_INV,D2,IR4,g,f)
 implicit none
 integer, intent(in) :: N_fm,nr
 real(kind=8), dimension(:,:), intent(in) :: IR4,D2
 real(kind=8), dimension(:,:,:), intent(in) :: A4_INV
 real(kind=8), dimension(:), intent(in) :: g
 real(kind=8), dimension(nr*N_fm), intent(out) :: f
 
 real(kind=8), dimension(nr) :: f_e,bf_e,b_test
 real(kind=8), dimension(nr,nr) :: L1;!,L;	
 real(kind=8) :: bj,bjt,j
 integer :: jj,row,ind_j;
 f = 0.d0; 
 
 
 !~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~##~~~~~~~~~~~~~~~~~~~~~
 f_e = 0.d0; bf_e = 0.d0; b_test = 0.d0; ! Do ODDS
 bj = 0.d0; bjt = 0.d0;

 do jj = 0,(N_fm-1),2

  row = (N_fm - (jj + 1) ); 
  ind_j = row*nr; !# Row ind
  j = float(N_fm-jj); ! Remeber sine counting is from 1 - N_theta
  bj = -j*(j + 1.d0); bjt = -2.d0*j;
  
  L1 = D2 + bj*IR4; 
  !L = D4 + bj*L1;
    
  IF (row < (N_fm - 2 ) ) THEN
   
   f_e = f_e + f( 1+ (row+2)*nr:(row+3)*nr);

   b_test = bjt*( matmul(L1, f_e ) + matmul(IR4, bf_e) );
   
   !f(1+ ind_j:ind_j+nr) = Ax(-L,g(1+ind_j:ind_j+nr)+b_test,nr);   !#O(Nr^3 N_theta)

   f(1+ ind_j:ind_j+nr) = matmul(A4_inv(:,:,1+jj),g(1+ind_j:ind_j+nr)+b_test);

   !# Add sums after to get +2 lag
   bf_e = bf_e + bj*f( 1+ ind_j:ind_j+nr) + bjt*f_e;	
  ELSE
   !f(1+ ind_j:ind_j+nr) = Ax(-L,g(1+ind_j:ind_j+nr),nr);
   f(1+ ind_j:ind_j+nr) = matmul(A4_inv(:,:,1+jj),g(1+ind_j:ind_j+nr) );
   bf_e = bf_e + bj*f( 1+ ind_j:ind_j+nr);
  END IF

 end do 

 
 !~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~##~~~~~~~~~~~~~~~~~~~~~
 f_e = 0.d0; bf_e = 0.d0; b_test = 0.d0; ! Do EVENS
 bj = 0.d0; bjt = 0.d0;

 do jj = 1,(N_fm-1),2

  row = (N_fm - (jj + 1) ); 
  ind_j = row*nr; !# Row ind
  j = float(N_fm-jj); ! Remeber sine counting is from 1 - N_theta
  bj = -j*(j + 1.d0); bjt = -2.d0*j;
  
  L1 = D2 + bj*IR4; 
  !L = D4 + bj*L1;
    
  IF (row < (N_fm - 2 ) ) THEN
   
   f_e = f_e + f( 1+ (row+2)*nr:(row+3)*nr);

   b_test = bjt*( matmul(L1, f_e ) + matmul(IR4, bf_e) );
   
   !f(1+ ind_j:ind_j+nr) = Ax(-L,g(1+ind_j:ind_j+nr)+b_test,nr);   !#O(Nr^3 N_theta)

   f(1+ ind_j:ind_j+nr) = matmul(A4_inv(:,:,1+jj),g(1+ind_j:ind_j+nr)+b_test);

   !# Add sums after to get +2 lag
   bf_e = bf_e + bj*f( 1+ ind_j:ind_j+nr) + bjt*f_e;	
  ELSE
   !f(1+ ind_j:ind_j+nr) = Ax(-L,g(1+ind_j:ind_j+nr),nr);
   f(1+ ind_j:ind_j+nr) = matmul(A4_inv(:,:,1+jj),g(1+ind_j:ind_j+nr) );
   bf_e = bf_e + bj*f( 1+ ind_j:ind_j+nr);
  END IF

 end do 
 
end subroutine A4_BSub_V2


!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~
! ***** Symmetric counterpatrts and initialize as global
!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~



!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~
! ***** Solves -Pr*A^4*f = g
!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~
subroutine A4_BSub_SYM_V2(N_fm,nr,A4_INV,D2,IR4,g,f)
 implicit none
 integer, intent(in) :: N_fm,nr
 real(kind=8), dimension(:,:), intent(in) :: IR4,D2
 real(kind=8), dimension(:,:,:), intent(in) :: A4_INV
 real(kind=8), dimension(:), intent(in) :: g
 real(kind=8), dimension(nr*N_fm), intent(out) :: f
 
 real(kind=8), dimension(nr) :: f_e,bf_e,b_test
 real(kind=8), dimension(nr,nr) :: L1;!,L;	
 real(kind=8) :: bj,bjt,j
 integer :: jj,row,ind_j;
 f = 0.d0; 
  
 !~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~##~~~~~~~~~~~~~~~~~~~~~
 f_e = 0.d0; bf_e = 0.d0; b_test = 0.d0; ! Do EVENS
 bj = 0.d0; bjt = 0.d0;

 do jj = 0,(N_fm-1),2

  row = (N_fm - (jj + 1) ); 
  ind_j = row*nr; !# Row ind
  j = float(N_fm-jj); ! Remeber sine counting is from 1 - N_theta
  bj = -j*(j + 1.d0); bjt = -2.d0*j;
  
  L1 = D2 + bj*IR4; 
  !L = D4 + bj*L1;
    
  IF (row < (N_fm - 2 ) ) THEN
   
   f_e = f_e + f( 1+ (row+2)*nr:(row+3)*nr);

   b_test = bjt*( matmul(L1, f_e ) + matmul(IR4, bf_e) );
   
   !f(1+ ind_j:ind_j+nr) = Ax(-L,g(1+ind_j:ind_j+nr)+b_test,nr);   !#O(Nr^3 N_theta)

   f(1+ ind_j:ind_j+nr) = matmul(A4_inv(:,:,1+jj),g(1+ind_j:ind_j+nr)+b_test);

   !# Add sums after to get +2 lag
   bf_e = bf_e + bj*f( 1+ ind_j:ind_j+nr) + bjt*f_e;	
  ELSE
   !f(1+ ind_j:ind_j+nr) = Ax(-L,g(1+ind_j:ind_j+nr),nr);
   f(1+ ind_j:ind_j+nr) = matmul(A4_inv(:,:,1+jj),g(1+ind_j:ind_j+nr) );
   bf_e = bf_e + bj*f( 1+ ind_j:ind_j+nr);
  END IF

 end do 
 
end subroutine A4_BSub_SYM_V2


! ***** Solves -Nabla^2*f = g
subroutine nab2_SYM_V2(N_fm,nr,N2_INV,g,f)
 implicit none
 integer, intent(in) :: N_fm,nr
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
 	
  
  ! 1+ due to fortran indexing
  f(1+ind_j:ind_j+nr) = matmul( N2_INV(:,:,1+jj) , g( 1+ ind_j:ind_j+nr)-b );	
	
 end do
 
end subroutine nab2_SYM_V2


subroutine T0_JT_SYM(IR,nr,N_fm,g,f)
 implicit none
 integer, intent(in) :: N_fm,nr
 real(kind=8), dimension(:), intent(in) :: IR
 real(kind=8), dimension(:), intent(in) :: g
 real(kind=8), dimension(nr*N_fm), intent(out) :: f
 
 real(kind=8), dimension(nr) :: b
 integer :: jj,ind_j,ind_k,j
 f = 0.d0; 
 
 !~~~~~~~~~~~~~~~~~~#
 b = 0.d0; ! Do Evens
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



subroutine JT_SYM(nr,N_fm,g,f)
 implicit none
 integer, intent(in) :: N_fm,nr
 real(kind=8), dimension(:), intent(in) :: g
 real(kind=8), dimension(nr*N_fm), intent(out) :: f
 
 real(kind=8), dimension(nr) :: b
 integer :: jj,ind_j,ind_k,j
 f = 0.d0;
 
 !~~~~~~~~~~~~~~~~~~#
 b = 0.d0; !Do EVENS
 do jj = 1,(N_fm-1),2

  j = (N_fm - (jj + 1) );
  ind_j = j*nr;
  ind_k = (j -1)*nr;
		
  IF (j < (N_fm - 1 ) ) THEN
    b = b + g( 1+ ind_k+2*nr:ind_k+3*nr);	
  END IF
		
  IF (j == 0) THEN
    f(1+ind_j:ind_j+nr) = b;
  ELSE
    f(1+ind_j:ind_j+nr) = float(j + 1)*g(1+ind_k:ind_k+nr) + 2.d0*b;	
  END IF

 end do
	
end subroutine JT_SYM
!~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~##~~~~~~~~~~~~~~~~~~~~~~~~

subroutine A2_SINE_SYM(D2,IR4,nr,N_fm,g,f)
 implicit none
 integer, intent(in) :: N_fm,nr
 real(kind=8), dimension(:,:), intent(in) :: D2,IR4
 real(kind=8), dimension(:), intent(in) :: g
 real(kind=8), dimension(nr*N_fm), intent(out) :: f
 
 real(kind=8), dimension(nr) :: f_e
 integer :: jj,ind_j,j
 f = 0.d0; !jj = 0; row = 0; ind_j = 0;
 
 !~~~~~~~~~~~~~~~~~~#
 f_e = 0.d0; !Do_EVENS
 do jj = 0,(N_fm-1),2

  j = N_fm-jj; 
  ind_j = (j-1)*nr; !# Row ind
							
  f(1+ ind_j:ind_j+nr) = matmul(D2,g(1+ind_j:ind_j+nr)) -float(j)*matmul(IR4,float((j+1))*g(1+ind_j:ind_j+nr)+ 2.d0*f_e);
  f_e = f_e + g(1+ind_j:ind_j+nr);
  
 end do
	
end subroutine A2_SINE_SYM


!~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~##~~~~~~~~~~~~~~~~~~~~~~~~
subroutine DerivTC(X,Dr,N_fm,nr,DT,kT,DC,kC)
 implicit none
 integer, intent(in) :: N_fm,nr
 real(kind=8), dimension(:), intent(in) :: X
 real(kind=8), dimension(:,:), intent(in) :: Dr
 
 integer :: ii, ind_T, ind_C, k_c, N;
 real(kind=8), dimension(nr) :: T,C	

 !# ADD space for padding to anti-aliase
 real(kind=8), dimension(nr,3*(N_fm/2)), intent(out) :: DT,kT,DC,kC

 N = nr*N_fm;	
 DT = 0.d0; kT = 0.d0; DC = 0.d0; kC = 0.d0;
 
 !!$OMP parallel do private(Nx,nr,j1,k1,ind_j,ind_k,A,X),reduction(+:Lx)
 !!$OMP end parallel do
	
 
 do ii = 1,N_fm ! Correct indexing is from 1
  k_c = ii-1; ![0,N_fm-1]
   
  ind_T = 1 + N + k_c*nr;  ! Row index

  T = X(ind_T:ind_T+nr-1);
  
  DT(:,ii) = matmul(Dr,T);
  kT(:,ii) = -dble(k_c)*T; ! Cosine -> Sine
 
  !# c) ~~~~~~~~~~ C parts ~~~~~~~~~~~~ # Correct

  ind_C = 1 + 2*N + k_c*nr;  ! Row index

  C = X(ind_C:ind_C+nr-1);
  
  DC(:,ii) = matmul(Dr,C);
  kC(:,ii) = -dble(k_c)*C; ! Cosine -> Sine 
 
 end do
 
end subroutine DerivTC


subroutine Deriv_psi(X,JPSI,OMEGA,Dr,N_fm,nr,Dpsi_h,kDpsi_h,JT_psi_h,DJT_psi_h,omega_h,komega_h,Domega_h)
 implicit none
 integer, intent(in) :: N_fm,nr
 real(kind=8), dimension(:), intent(in) :: X,JPSI,OMEGA
 real(kind=8), dimension(:,:), intent(in) :: Dr
 
 integer :: ii,ind,N,k_s;
 real(kind=8), dimension(nr) :: psi	

 !# ADD space for padding to anti-aliase
 real(kind=8), dimension(nr,3*(N_fm/2)), intent(out) :: Dpsi_h,kDpsi_h
 real(kind=8), dimension(nr,3*(N_fm/2)), intent(out) :: JT_psi_h,DJT_psi_h
 real(kind=8), dimension(nr,3*(N_fm/2)), intent(out) :: omega_h,komega_h,Domega_h

 Dpsi_h = 0.d0; kDpsi_h = 0.d0; 
 JT_psi_h = 0.d0; DJT_psi_h = 0.d0;
 omega_h = 0.d0; komega_h = 0.d0; Domega_h = 0.d0;

 N = nr*N_fm;	
 
 !!$OMP parallel do private(Nx,nr,j1,k1,ind_j,ind_k,A,X),reduction(+:Lx)
 !!$OMP end parallel do
	
 do ii = 1,N_fm ! Correct indexing is from 1
  k_s = ii; ![1,N_fm]
   
  ind = 1 + (ii-1)*nr;  ! Row index
  psi = X(ind:ind+nr-1);

  Dpsi_h(:,ii) = matmul(Dr,psi);
  kDpsi_h(:,ii) = k_s*Dpsi_h(:,ii); !# Sine -> Cosine #
				
  JT_psi_h(:,ii) = JPSI(ind:ind+nr-1);
  DJT_psi_h(:,ii) = matmul(Dr,JT_psi_h(:,ii)); 

  omega_h(:,ii) = OMEGA(ind:ind+nr-1);
  komega_h(:,ii) = k_s*omega_h(:,ii) !# Sine -> Cosine 

  Domega_h(:,ii) = matmul(Dr,omega_h(:,ii)); 
 
 end do
 
end subroutine Deriv_psi


! ########## SYMMETRIC FUNCTION ~~~~~~~~~~~~###########
function LX(N_fm,nr,Ra,Ra_s,GX,DT0,X)

 implicit none
 integer, intent(in) :: N_fm,nr
 real(kind=8), intent(in) :: Ra,Ra_s
 real(kind=8), dimension(:),intent(in) :: GX
 real(kind=8), dimension(:), intent(in) :: DT0
 real(kind=8), dimension(:),intent(in) :: X
 
 real(kind=8) :: LX(size(X)); 
 integer :: N; 
 N = nr*N_fm; LX = 0.d0;
 
 !print *, 'X=',X;
 !print *,'GX*X',GX*X(3:N-2)

 ! 1) Vorticity
 LX(1:N-nr) = GX*( Ra*X(1+nr+N:2*N) - Ra_s*X(1+nr+(2*N):3*N) );
 
 ! 2) Temperature
 call T0_JT_SYM(DT0,nr,N_fm,X(1:N), LX(1+N:2*N) );
 !LX(N:2*N) = t0_jt_sym(DT0,nr,N_fm,X(0:N));

 ! 3) Concentration
 LX(1+2*N:3*N) = LX(1+N:2*N)

end function LX;


subroutine PRECOND_MFREE_SYM(N_fm,nr,Tau, Ra,Ra_s,GX,DT0, N2_INV, A4_INV,D2,IR4,NX,X,AX)

 implicit none
 integer, intent(in) :: N_fm,nr;
 real(kind=8)::Tau,Ra,Ra_s;
 real(kind=8), dimension(:), intent(in) :: GX,DT0
 real(kind=8), dimension(:,:,:), intent(in) :: N2_INV, A4_INV;
 real(kind=8), dimension(:,:), intent(in) :: D2,IR4;
 real(kind=8), dimension(:), intent(in) :: NX,X
 
 real(kind=8),dimension(3*nr*N_fm,1),intent(out)::AX;
 integer :: N; !jj
 real(kind=8),dimension(3*nr*N_fm):: FX	
 
 AX = 0.d0; FX = 0.d0; N = nr*N_fm;	
	
 ! A) Update NX to have linear part added	
 ! ~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~
 FX = LX(N_fm,nr,Ra,Ra_s,GX,DT0,X) + NX;
 
 ! B) Apply preconditioner in parallel
 ! ~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~
 
 !~~~~~~~~~~

 ! 1) Vorticity
 call a4_bsub_sym_v2(N_fm,nr,A4_INV,D2,IR4,FX(1:N), AX(1:N,1) )

 ! 2) Temperature
 call nab2_sym_v2(N_fm,nr,N2_INV,FX(1+N:2*N), AX(1+N:2*N,1) );

 !3) Concentration 
 call nab2_sym_v2(N_fm,nr,N2_INV,FX(1+ 2*N:3*N),AX(1+ 2*N:3*N,1) );

 AX(1:N,1) = AX(1:N,1) - X(1:N);
 AX(1+N:2*N,1) = AX(1+N:2*N,1) - X(1+N:2*N);
 AX(1+2*N:3*N,1) = (1./Tau)*AX(1+2*N:3*N,1) - X(1+2*N:3*N);
 
 !~~~~~~~~~~

end subroutine PRECOND_MFREE_SYM

!~#############################################################

!#~~~~~~~~~~ SYMMETRIC VERSION #~~~~~~~~~~~~~~~~~~~~
!~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~##~~~~~~~~~~~~~~~~~~~~~~~~
subroutine DerivTC_SYM(X,Dr,N_fm,nr,DT,kT,DC,kC)

 implicit none
 integer, intent(in) :: N_fm,nr
 real(kind=8), dimension(:), intent(in) :: X
 real(kind=8), dimension(:,:), intent(in) :: Dr
 
 integer :: ii, ind_T, k_c, N;
 real(kind=8), dimension(nr) :: T,C	

 !# ADD space for padding to anti-aliase
 real(kind=8), dimension(nr,3*(N_fm/2)), intent(out) :: DT,kT,DC,kC

 T = 0.d0; C = 0.d0;
 N = nr*N_fm;	
 DT = 0.d0; kT = 0.d0; DC = 0.d0; kC = 0.d0;
 	
 do ii = 1,N_fm,2 ! Correct indexing is from 1
  k_c = ii-1; ![0,N_fm-1]
  ind_T = 1 + N + k_c*nr;  ! Row index

  T = X(ind_T:ind_T+nr-1);
  C = X(ind_T + N:ind_T + N +nr-1);	

  !IF (MOD(k_c,2) == 0) THEN
  DT(:,ii) = matmul(Dr,T);
  DC(:,ii) = matmul(Dr,C);
  kT(:,ii) = -dble(k_c)*T; ! Cosine -> Sine
  kC(:,ii) = -dble(k_c)*C; ! Cosine -> Sine 
  !END IF

 end do
 
end subroutine DerivTC_SYM


subroutine Deriv_psi_SYM(X,JPSI,OMEGA,Dr,N_fm,nr,Dpsi_h,kDpsi_h,JT_psi_h,DJT_psi_h,omega_h,komega_h,Domega_h)
 
 implicit none
 integer, intent(in) :: N_fm,nr
 real(kind=8), dimension(:), intent(in) :: X,JPSI,OMEGA
 real(kind=8), dimension(:,:), intent(in) :: Dr
 
 integer :: ii,ind,N,k_s; !!DeN_fm;
 real(kind=8), dimension(nr) :: psi	

 !!DeN_fm = int(3*(N_fm/2));	
 !# ADD space for padding to anti-aliase
 real(kind=8), dimension(nr,3*(N_fm/2)), intent(out) :: Dpsi_h,kDpsi_h
 real(kind=8), dimension(nr,3*(N_fm/2)), intent(out) :: JT_psi_h,DJT_psi_h
 real(kind=8), dimension(nr,3*(N_fm/2)), intent(out) :: omega_h,komega_h,Domega_h

 psi = 0.d0
 Dpsi_h = 0.d0; kDpsi_h = 0.d0; 
 JT_psi_h = 0.d0; DJT_psi_h = 0.d0;
 omega_h = 0.d0; komega_h = 0.d0; Domega_h = 0.d0;

 N = nr*N_fm;	
 
 !!$OMP parallel do private(Nx,nr,j1,k1,ind_j,ind_k,A,X),reduction(+:Lx)
 !!$OMP end parallel do

 ! Do EVENS ONLY	
 do ii = 1,N_fm ! Correct indexing is from 1
  k_s = ii; ![1,N_fm]
   
  ind = 1 + (ii-1)*nr;  ! Row index
  psi = X(ind:ind+nr-1);

  IF (mod(k_s,2) == 0) THEN
   Dpsi_h(:,ii) = matmul(Dr,psi);
   kDpsi_h(:,ii) = dble(k_s)*Dpsi_h(:,ii); !# Sine -> Cosine #
				
   omega_h(:,ii) = OMEGA(ind:ind+nr-1);
   komega_h(:,ii) = dble(k_s)*omega_h(:,ii) !# Sine -> Cosine 

   Domega_h(:,ii) = matmul(Dr,omega_h(:,ii)); 
 ELSE IF (mod(k_s,2) == 1) THEN
  JT_psi_h(:,ii) = JPSI(ind:ind+nr-1);
  DJT_psi_h(:,ii) = matmul(Dr,JT_psi_h(:,ii));  
 END IF

 end do
 
end subroutine Deriv_psi_SYM


subroutine Reshape_Nx(J_PSI,J_PSI_T,J_PSI_C,aks,akc,N_fm,nr,Nx)
 
 implicit none
 integer, intent(in) :: N_fm,nr
 real(kind=8), dimension(:,:), intent(in) :: J_PSI,J_PSI_T,J_PSI_C
 real(kind=8), dimension(:), intent(in) :: aks,akc
 real(kind=8), dimension(3*N_fm*nr), intent(out) :: Nx
 
 integer :: ind,ind_T,ind_C
 integer :: ii,N
 Nx = 0.d0; ind= 0; N = N_fm*nr;
 
 !!$OMP parallel do private(Nx,nr,j1,k1,ind_j,ind_k,A,X),reduction(+:Nx)
 do ii = 1,N_fm

  ind = 1 + (ii-1)*nr;  ! Row index 	
  Nx(ind:ind+nr-1) = aks(ii)*J_PSI(:,ii);
  
  ind_T = 1 + N + (ii-1)*nr;  ! Row index 	
  Nx(ind_T:ind_T+nr-1) = akc(ii)*J_PSI_T(:,ii);

  ind_C = 1 + 2*N + (ii-1)*nr;  ! Row index 		
  Nx(ind_C:ind_C+nr-1) = akc(ii)*J_PSI_C(:,ii);	  
 
 end do
 !!$OMP end parallel do
end subroutine Reshape_Nx


end module NON_LIN1
