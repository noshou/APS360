!> @brief Complete form factor calculation for X-ray scattering
!>
!! @details
!! Computes the complete atomic form factor by combining the atomic form
!! factor f0(Q) with anomalous scattering corrections f1 and f2:
!!
!!   getFormFact(Q, elm) = f0(Q, elm) + f1(elm) + i*f2(elm)
!!
!! where:
!!   - f0(Q) is the normal atomic scattering factor (Q-dependent)
!!   - f1 is the real anomalous correction (dispersion correction)
!!   - f2 is the imaginary anomalous correction (absorption)
module FormFact
    use iso_c_binding, only: c_double
    use F0Factor
    use F1F2Factors
    implicit none
    
    private
    public :: getFormFact, getQValues
    public :: getFormFactPy
    
contains

    !! Computes the complex atomic form factor by combining:
    !!   - f0(Q): Q-dependent atomic scattering factor
    !!   - f1: Real anomalous scattering correction
    !!   - f2: Imaginary anomalous scattering correction
    !!
    !! Result: ff = (f0 + f1) + i*f2
    !! 
    !! If element is not found, returns (0,0) and status = -1
    !! @param[in]  q      Scattering vector magnitude (sin θ)/λ in Å⁻¹
    !! @param[in]  elm    Element symbol (case-insensitive, e.g., 'Fe', 'Cu')
    !! @param[out] status Return status: 0 = success, -1 = element not found, -2 = Q out of range
    !!
    !! @return ff Complex form factor
    function getFormFact(q, elm, status) result(ff)
        real(c_double), intent(in) :: q
        character(len=*), intent(in) :: elm
        integer, intent(out) :: status
        complex(c_double) :: ff

        real(c_double) :: f0Val, f1, f2, f0_0

        if (q < 0.0_c_double .or. q > 0.5_c_double) then
            ff = cmplx(0.0_c_double, 0.0_c_double, kind=c_double)
            status = -2
            return
        end if

        ! Get anomalous scattering factors f1 and f2
        call getF1F2(elm, f1, f2, status)

        if (status /= 0) then
            ff = cmplx(0.0_c_double, 0.0_c_double, kind=c_double)
            return
        end if

        ! Get atomic form factor f0
        f0Val = getF0(q, elm)
        
        ! get f0(Q=0)
        f0_0  = getF0(0.0_c_double, elm)

        ! Construct complex result: real = f1 + f0 - f0(Q), imaginary = f2
        ff = cmplx(f1 + f0Val - f0_0, f2, kind=c_double)
        
    end function getFormFact

    !> Wrapper for python since it struggled returning compelx numbers
    subroutine getFormFactPy(q, elm, ff_re, ff_im, status)
        real(c_double), intent(in)  :: q
        character(len=*), intent(in) :: elm
        real(c_double), intent(out) :: ff_re, ff_im
        integer, intent(out) :: status
        complex(c_double) :: ff
        ff = getFormFact(q, elm, status)
        ff_re = real(ff, kind=c_double)
        ff_im = aimag(ff)
    end subroutine getFormFactPy

    !! @return the list of available q values
    function getQValues() result(qvals)
        real(c_double), allocatable :: qvals(:)
        qvals = getQVals()
    end function getQValues
end module FormFact