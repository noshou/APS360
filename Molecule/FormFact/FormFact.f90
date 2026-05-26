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
    use F0Factor, only: getF0, getQVals
    use F1F2Factors, only: getF1F2
    implicit none
    
    private
    public :: getFormFact, getQValues
    
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
    !! @param[out] status Return status: 0 = success, -1 = element not found
    !! 
    !! @return ff Complex form factor
    function getFormFact(q, elm, status) result(ff)
        real(c_double), intent(in) :: q
        character(len=*), intent(in) :: elm
        integer, intent(out) :: status
        complex(c_double) :: ff
        
        real(c_double) :: f0Val, f1, f2
        
        ! Get anomalous scattering factors f1 and f2
        call getF1F2(elm, f1, f2, status)
        
        if (status /= 0) then
            ! Element not found, return zero and error status
            ff = cmplx(0.0_c_double, 0.0_c_double, kind=c_double)
            return
        end if
        
        ! Get atomic form factor f0
        f0Val = getF0(q, elm)
        
        ! Construct complex result: real = f1 + f0, imaginary = f2
        ff = cmplx(f1 + f0Val, f2, kind=c_double)
        
    end function getFormFact

    !! @return the list of available q values
    function getQValues() result(qvals)
        real(c_double), allocatable :: qvals(:)
        allocate(qvals, source=getQVals())
    end function getQValues

end module FormFact