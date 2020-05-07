! ~~~~ f2py -c --fcompiler=intelem --opt='-O3' --f90flags='-fopenmp' -liomp5 F2PY_Module_N1_TSTEP_PARA.f90 -m N1_FT_PARA ~~~~ To compile

! ~~~~ f2py -c --fcompiler=gnu95 --opt='-O3' --f90flags='-fopenmp' -lgomp F2PY_Module_N1_TSTEP_PARA.f90 -m N1_FT_PARA ~~~~ To compile

! For gfortran f2py -c file.f90 -m file_module --f90flags='-fopenmp' -lgomp
! for ifort f2py -c file.f90 -m file_module --fcompiler=intelem --f90flags='-qopenmp' -liomp5

module NON_LIN1
 !use types, only : dp, pi
 !use utils, only : stop_error
 use omp_lib
 use iso_c_binding
 !!!makes OpenMP routines, variables available
 implicit none
 integer :: i1 !, j1, k1, iu
 integer :: NumThreads,threadID
 !integer :: N_modes, nr
 !integer :: l,m,n,ind_n,ind_m
 real(kind=8), dimension(31,31,31) :: aPSI,bPSI,aT,bT
 save
!private
contains

!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~
! ***** Initialize all coefficient arrays
!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~
subroutine initialize_coeffs()
 implicit none

 !open(1, file='aPsi.unf', form='unformatted')
 !read(1) aPSI
 !close(1)
 open(40, file='aPSI.bin', status='old', access='stream', form='unformatted')
 read(40) aPSI
 close(40)
 open(40, file='bPSI.bin', status='old', access='stream', form='unformatted')
 read(40) bPSI
 close(40)
 open(40, file='aT.bin', status='old', access='stream', form='unformatted')
 read(40) aT
 close(40)
 open(40, file='bT.bin', status='old', access='stream', form='unformatted')
 read(40) bT
 close(40)

 !print *,'aT()',aT(1,:,2) !(1+1,2+1,1+1)
 !print *,'aT(1,:,1)',aT(2,:,2) !(1+1,2+1,1+1)
 !print *,'aT(2,:,1)',aT(3,:,2) !(1+1,2+1,1+1)

 !print *,'aPSI()',aPSI(9+1,5+1,1+1)
 !print *,'bPSI()',bPSI(3+1,6+1,7+1)
 !print *,'aT()',aT(2+1,3+1,2+1)
 !print *,'bT()',bT(3+1,4+1,1+1)
 
end subroutine initialize_coeffs


!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~
! ***** Generate R,DB + Maybe D2,D3 ~~~ and initialize as global
!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~




!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~
! ***** Function for x |-> A on diagonal
!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~
function diag(X1)
 real(kind=8), dimension(:) :: X1
 real(kind=8):: diag(size(X1),size(X1))
 diag = 0.d0
 do i1 = 1,size(X1) 
  diag(i1,i1) = X1(i1) 
 end do
end function diag

!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~
! ***** Subroutine to compute N2_PSI_T_OM
!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~
subroutine N1_PSI(Psi_0,DB,D2,D3,aPSI,bPSI,aT,bT,A1,A2)

 real(kind=8), dimension(:), intent(in) :: Psi_0
 real(kind=8), dimension(:,:),intent(in) :: DB,D2,D3
 real(kind=8), intent(in) :: aPSI,bPSI,aT,bT

 real(kind=8), dimension(size(Psi_0),size(Psi_0)), intent(out) :: A1,A2 ! function output
 real(kind=8), dimension(size(Psi_0),size(Psi_0)) :: PSI_D,D_PSI,PSI
 real(kind=8), dimension(size(Psi_0),size(Psi_0)) :: PSI_D3M,DPSI_D2M 
 
 A1 = 0.d0; A2 = 0.d0;
 D_PSI=0.d0; PSI_D=0.d0; PSI=0.d0;
 PSI_D3M=0.d0; DPSI_D2M=0.d0; 

 PSI = diag(Psi_0);
 PSI_D = matmul(PSI,DB);
 D_PSI = diag(matmul(DB,Psi_0));
	
 PSI_D3M = matmul(PSI,D3);
 DPSI_D2M = matmul(D_PSI,D2)

 A1 = dble(aPSI)*PSI_D3M + dble(bPSI)*DPSI_D2M;
 A2 = dble(aT)*PSI_D + dble(bT)*D_PSI;

 ! deallocate stuff????

end subroutine N1_PSI




!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~
! ***** Sublock
!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~
function N1_n(X,DB,D2,D3,R,aPSI,bPSI,aT,bT,l)

 integer, intent(in) :: l
 real(kind=8), dimension(:), intent(in) :: X
 real(kind=8), dimension(:), intent(in) :: R
 real(kind=8), dimension(:,:),intent(in) :: DB,D2,D3
 real(kind=8), intent(in) :: aPSI,bPSI,aT,bT
 real(kind=8) :: N1_n(3*size(R),3*size(R))
 
 integer :: nr,ind
 real(kind=8), dimension(size(R)) :: Psi_0
 real(kind=8), dimension(size(R),size(R)) :: A1,A2

 ! Always necessary / good practice ?????
 N1_n = 0.d0; A1 = 0.d0; A2 = 0.d0; Psi_0 = 0.d0;
 
 nr = size(R)

 IF (l .GT. 0) THEN ! Condition used as all coefficients contain l*(l+1), and that \psi_{l=0} = 0
  
  ind= 1 + l*3*nr;  ! (+1) fortan index
  Psi_0 = X(ind:ind+nr-1)
  call N1_PSI(Psi_0,DB,D2,D3,aPSI,bPSI,aT,bT,A1,A2)
  
  N1_n(1:nr,1:nr) = A1; !# N_PSI
  N1_n(1+nr:2*nr,1+nr:2*nr) = A2;  !# N_T
  N1_n(1+2*nr:3*nr,1+2*nr:3*nr) = A2 !# N_C == N_T
 
 END IF

 ! deallocate stuff????
end function N1_n



!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~
!**************** FULL BLOCK **********************
!# ~~~~~~~~~~~~~~~~~~~# #~~~~~~~~~~~~~~~~~~~ #~~~~~~~~~~~~~~~~~~~~~~~~~~~
!subroutine N1_Full(X,D3_F,D2_F,DB,R,aPSI,bPSI,aT,bT,N_F)
subroutine N1_Full(X,D3_F,D2_F,DB,R,N_F)

 real(kind=8), dimension(:), intent(in) :: X
 real(kind=8), dimension(:,:,:),intent(in) :: D3_F,D2_F
 real(kind=8), dimension(:,:),intent(in) :: DB
 real(kind=8), dimension(:), intent(in) :: R
 !!!! Passing ~100Mb in!!!! TOO SLOW, USE COMMON BLOCK
 !real(kind=8), dimension(100,100,100), intent(in) :: aPSI,bPSI,aT,bT
 real(kind=8), dimension(size(X)), intent(out) :: N_F ! LARGE Vector TO CREATE

 integer :: l,m,n,ind_n,ind_m, nr, N_modes
 real(kind=8), dimension(size(R),size(R)) :: D2,D3
 real(kind=8), dimension(3*size(R),3*size(R)) :: A_block 
 real(kind=8) :: A_PSI,B_PSI,A_T,B_T

 D2=0.d0; D3=0.d0; A_block = 0.d0; N_F = 0.d0
 A_PSI=0.d0; B_PSI =0.d0; A_T =0.d0; B_T =0.d0;
 
 nr = int(size(R)); N_modes = int(size(X)/(3*nr));

 !!$OMP PARALLEL PRIVATE(threadID)
 ! NumThreads = omp_get_num_threads()
 ! threadID = omp_get_thread_num()
 ! !print *, 'this is thread',threadID, ' of ', NumThreads
 !!$OMP END PARALLEL

 !!$ call omp_set_num_threads(2)
 
 ! The ! followed by $ Starts OMP Directive 
 ! Statement which follows starts Parallel region
 ! Each thread will have a copy of the private variables, and iterator (l) is auto private
 !$OMP parallel do private(n,m,ind_n,ind_m,A_PSI,B_PSI,A_T,B_T,D2,D3,A_block),reduction(+:N_F)
 do l = 0,(N_modes-1) ! mode number not index 0 -> 30
    do n = 0,(N_modes-1)
      ind_n = 1 + n*3*nr

      do m = 0,(N_modes-1)
        ind_m = 1 + m*3*nr
        
         IF (mod(l+m+n,2) .EQ. 0) THEN
               !IF ( (abs(l-m) .LE. n) .AND. (n .LE. abs(l+m)) ) THEN
               IF ( (l .LE. (m+n)) .AND. (m .LE. (l+n)) .AND. (n .LE. (l+m)) ) THEN
                
		! Obtain all coefficients, index is (+1) due to fortran
                A_PSI = aPSI(l+1,m+1,n+1); B_PSI = bPSI(l+1,m+1,n+1)
                A_T = aT(l+1,m+1,n+1); B_T = bT(l+1,m+1,n+1);
                
                ! Obtain D2,D3, index is (+1) due to fortran
                D2 = D2_F(:,:,m+1); D3 = D3_F(:,:,m+1); ! Mode m, as Acts on vector X_m
                A_block = N1_n(X,DB,D2,D3,R,A_PSI,B_PSI,A_T,B_T,l); ! Mode l, as takes and processes X_l
                N_F(ind_n:ind_n+3*nr-1) = N_F(ind_n:ind_n+3*nr-1) + matmul(A_block,X(ind_m:ind_m+3*nr-1) ); ! Computes X_n = X_n + <X_l=0,X_{m=0,...,N_modes -1} >
               END IF
         END IF
      end do 
    end do
 end do
 !$OMP end parallel do
 ! ENDS OMP Directive & Parallel region	
end subroutine N1_Full


end module NON_LIN1
