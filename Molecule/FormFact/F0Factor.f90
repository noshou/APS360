!> Atomic form factors f0 for X-ray scattering calculations
!>
!! Atomic form factors f0 for X-ray scattering calculations.
!! f0(Q) represents the scattering amplitude of an atom as a function
!! of the scattering vector Q = (sin θ)/λ.
!!
!! Data Source:
!!   International Tables for Crystallography Vol. C
!!   DOI: 10.1107/97809553602060000600
!!
module F0Factor
    use iso_c_binding, only: c_double
    implicit none

    private

    ! Public interface
    public :: nElements, nQValues
    public :: getF0, getQVals

    !---------------------------------------------------------------------------
    ! Module Data
    !---------------------------------------------------------------------------

    !> Number of elements in lookup table
    integer, parameter :: nElements = 25
    !> Number of Q values in lookup table
    integer, parameter :: nQValues = 26

    !> Element/ion symbols (lowercase)
    character(len=4), parameter :: elements(25) = [ &
            'ni  ', &
            'mg2+', &
            'se  ', &
            'cu2+', &
            'mn2+', &
            'ca  ', &
            'k   ', &
            'ar  ', &
            'cl  ', &
            's   ', &
            'p   ', &
            'si  ', &
            'al  ', &
            'mg  ', &
            'na  ', &
            'ne  ', &
            'f   ', &
            'o   ', &
            'n   ', &
            'c   ', &
            'b   ', &
            'be  ', &
            'li  ', &
            'he  ', &
            'h   ' ]

    !> Scattering vector magnitudes: Q = (sin θ)/λ in Å⁻¹
    !! Range: 0 to ~2.0 Å⁻¹ in increments of 0.01 Å⁻¹
    real(c_double), parameter :: qValues(26) = [ &
            0.000000_c_double, &
            0.020000_c_double, &
            0.040000_c_double, &
            0.060000_c_double, &
            0.080000_c_double, &
            0.100000_c_double, &
            0.120000_c_double, &
            0.140000_c_double, &
            0.160000_c_double, &
            0.180000_c_double, &
            0.200000_c_double, &
            0.220000_c_double, &
            0.240000_c_double, &
            0.260000_c_double, &
            0.280000_c_double, &
            0.300000_c_double, &
            0.320000_c_double, &
            0.340000_c_double, &
            0.360000_c_double, &
            0.380000_c_double, &
            0.400000_c_double, &
            0.420000_c_double, &
            0.440000_c_double, &
            0.460000_c_double, &
            0.480000_c_double, &
            0.500000_c_double ]

    !> Form factor data: f0(Q) for each element
    !! Rows: Q values, Columns: Elements
    !! f0 decreases with increasing Q due to destructive interference
    real(c_double), parameter :: f0Data(26,25) = reshape([ &
            28.000000_c_double, &
            10.000000_c_double, &
            34.000000_c_double, &
            27.000000_c_double, &
            23.000000_c_double, &
            20.000000_c_double, &
            19.000000_c_double, &
            18.000000_c_double, &
            17.000000_c_double, &
            16.000000_c_double, &
            15.000000_c_double, &
            14.000000_c_double, &
            13.000000_c_double, &
            12.000000_c_double, &
            11.000000_c_double, &
            10.000000_c_double, &
            9.000000_c_double, &
            8.000000_c_double, &
            7.000000_c_double, &
            6.000000_c_double, &
            5.000000_c_double, &
            4.000000_c_double, &
            3.000000_c_double, &
            2.000000_c_double, &
            1.000000_c_double, &
            27.892000_c_double, &
            9.986000_c_double, &
            33.881000_c_double, &
            26.956000_c_double, &
            22.953000_c_double, &
            19.838000_c_double, &
            18.854000_c_double, &
            17.924000_c_double, &
            16.919000_c_double, &
            15.915000_c_double, &
            14.914000_c_double, &
            13.900000_c_double, &
            12.903000_c_double, &
            11.914000_c_double, &
            10.922000_c_double, &
            9.973000_c_double, &
            8.970000_c_double, &
            7.967000_c_double, &
            6.963000_c_double, &
            5.958000_c_double, &
            4.954000_c_double, &
            3.950000_c_double, &
            2.947000_c_double, &
            1.993000_c_double, &
            0.991000_c_double, &
            27.579000_c_double, &
            9.945000_c_double, &
            33.532000_c_double, &
            26.824000_c_double, &
            22.812000_c_double, &
            19.392000_c_double, &
            18.462000_c_double, &
            17.700000_c_double, &
            16.683000_c_double, &
            15.665000_c_double, &
            14.646000_c_double, &
            13.628000_c_double, &
            12.629000_c_double, &
            11.674000_c_double, &
            10.709000_c_double, &
            9.891000_c_double, &
            8.881000_c_double, &
            7.869000_c_double, &
            6.855000_c_double, &
            5.837000_c_double, &
            4.820000_c_double, &
            3.807000_c_double, &
            2.802000_c_double, &
            1.972000_c_double, &
            0.966000_c_double, &
            27.096000_c_double, &
            9.876000_c_double, &
            32.982000_c_double, &
            26.608000_c_double, &
            22.581000_c_double, &
            18.758000_c_double, &
            17.924000_c_double, &
            17.340000_c_double, &
            16.306000_c_double, &
            15.271000_c_double, &
            14.237000_c_double, &
            13.209000_c_double, &
            12.222000_c_double, &
            11.319000_c_double, &
            10.412000_c_double, &
            9.757000_c_double, &
            8.736000_c_double, &
            7.712000_c_double, &
            6.682000_c_double, &
            5.645000_c_double, &
            4.613000_c_double, &
            3.592000_c_double, &
            2.606000_c_double, &
            1.939000_c_double, &
            0.925000_c_double, &
            26.490000_c_double, &
            9.782000_c_double, &
            32.273000_c_double, &
            26.311000_c_double, &
            22.266000_c_double, &
            18.045000_c_double, &
            17.332000_c_double, &
            16.865000_c_double, &
            15.814000_c_double, &
            14.764000_c_double, &
            13.721000_c_double, &
            12.695000_c_double, &
            11.739000_c_double, &
            10.903000_c_double, &
            10.084000_c_double, &
            9.576000_c_double, &
            8.541000_c_double, &
            7.501000_c_double, &
            6.453000_c_double, &
            5.396000_c_double, &
            4.352000_c_double, &
            3.336000_c_double, &
            2.400000_c_double, &
            1.893000_c_double, &
            0.872000_c_double, &
            25.807000_c_double, &
            9.662000_c_double, &
            31.449000_c_double, &
            25.939000_c_double, &
            21.875000_c_double, &
            17.331000_c_double, &
            16.733000_c_double, &
            16.298000_c_double, &
            15.234000_c_double, &
            14.177000_c_double, &
            13.138000_c_double, &
            12.134000_c_double, &
            11.230000_c_double, &
            10.472000_c_double, &
            9.760000_c_double, &
            9.351000_c_double, &
            8.302000_c_double, &
            7.245000_c_double, &
            6.180000_c_double, &
            5.107000_c_double, &
            4.060000_c_double, &
            3.065000_c_double, &
            2.215000_c_double, &
            1.837000_c_double, &
            0.811000_c_double, &
            25.083000_c_double, &
            9.519000_c_double, &
            30.557000_c_double, &
            25.500000_c_double, &
            21.418000_c_double, &
            16.655000_c_double, &
            16.138000_c_double, &
            15.665000_c_double, &
            14.597000_c_double, &
            13.546000_c_double, &
            12.527000_c_double, &
            11.567000_c_double, &
            10.733000_c_double, &
            10.059000_c_double, &
            9.455000_c_double, &
            9.090000_c_double, &
            8.026000_c_double, &
            6.954000_c_double, &
            5.875000_c_double, &
            4.794000_c_double, &
            3.756000_c_double, &
            2.804000_c_double, &
            2.065000_c_double, &
            1.774000_c_double, &
            0.744000_c_double, &
            24.344000_c_double, &
            9.335000_c_double, &
            29.637000_c_double, &
            25.001000_c_double, &
            20.904000_c_double, &
            16.024000_c_double, &
            15.543000_c_double, &
            14.991000_c_double, &
            13.932000_c_double, &
            12.902000_c_double, &
            11.922000_c_double, &
            11.025000_c_double, &
            10.273000_c_double, &
            9.678000_c_double, &
            9.166000_c_double, &
            8.799000_c_double, &
            7.721000_c_double, &
            6.637000_c_double, &
            5.551000_c_double, &
            4.472000_c_double, &
            3.459000_c_double, &
            2.569000_c_double, &
            1.950000_c_double, &
            1.701000_c_double, &
            0.676000_c_double, &
            23.604000_c_double, &
            9.171000_c_double, &
            28.718000_c_double, &
            24.451000_c_double, &
            20.344000_c_double, &
            15.430000_c_double, &
            14.941000_c_double, &
            14.301000_c_double, &
            13.263000_c_double, &
            12.270000_c_double, &
            11.345000_c_double, &
            10.525000_c_double, &
            9.857000_c_double, &
            9.334000_c_double, &
            8.888000_c_double, &
            8.483000_c_double, &
            7.395000_c_double, &
            6.304000_c_double, &
            5.218000_c_double, &
            4.153000_c_double, &
            3.179000_c_double, &
            2.365000_c_double, &
            1.863000_c_double, &
            1.624000_c_double, &
            0.608000_c_double, &
            22.872000_c_double, &
            8.969000_c_double, &
            27.822000_c_double, &
            23.857000_c_double, &
            19.748000_c_double, &
            14.859000_c_double, &
            14.334000_c_double, &
            13.615000_c_double, &
            12.611000_c_double, &
            11.668000_c_double, &
            10.811000_c_double, &
            10.074000_c_double, &
            9.487000_c_double, &
            9.023000_c_double, &
            8.613000_c_double, &
            8.150000_c_double, &
            7.055000_c_double, &
            5.964000_c_double, &
            4.886000_c_double, &
            3.847000_c_double, &
            2.924000_c_double, &
            2.197000_c_double, &
            1.796000_c_double, &
            1.543000_c_double, &
            0.542000_c_double, &
            22.150000_c_double, &
            8.751000_c_double, &
            26.962000_c_double, &
            23.229000_c_double, &
            19.126000_c_double, &
            14.304000_c_double, &
            13.728000_c_double, &
            12.949000_c_double, &
            11.991000_c_double, &
            11.109000_c_double, &
            10.327000_c_double, &
            9.673000_c_double, &
            9.158000_c_double, &
            8.735000_c_double, &
            8.335000_c_double, &
            7.805000_c_double, &
            6.709000_c_double, &
            5.623000_c_double, &
            4.563000_c_double, &
            3.560000_c_double, &
            2.699000_c_double, &
            2.060000_c_double, &
            1.742000_c_double, &
            1.460000_c_double, &
            0.481000_c_double, &
            21.438000_c_double, &
            8.521000_c_double, &
            26.145000_c_double, &
            22.574000_c_double, &
            18.488000_c_double, &
            13.760000_c_double, &
            13.130000_c_double, &
            12.315000_c_double, &
            11.413000_c_double, &
            10.598000_c_double, &
            9.894000_c_double, &
            9.319000_c_double, &
            8.862000_c_double, &
            8.465000_c_double, &
            8.052000_c_double, &
            7.454000_c_double, &
            6.362000_c_double, &
            5.289000_c_double, &
            4.254000_c_double, &
            3.297000_c_double, &
            2.503000_c_double, &
            1.951000_c_double, &
            1.693000_c_double, &
            1.377000_c_double, &
            0.424000_c_double, &
            20.737000_c_double, &
            8.280000_c_double, &
            25.372000_c_double, &
            21.900000_c_double, &
            17.517000_c_double, &
            13.225000_c_double, &
            12.550000_c_double, &
            11.721000_c_double, &
            10.881000_c_double, &
            10.138000_c_double, &
            9.510000_c_double, &
            9.004000_c_double, &
            8.592000_c_double, &
            8.205000_c_double, &
            7.764000_c_double, &
            7.102000_c_double, &
            6.020000_c_double, &
            4.965000_c_double, &
            3.963000_c_double, &
            3.058000_c_double, &
            2.336000_c_double, &
            1.864000_c_double, &
            1.648000_c_double, &
            1.295000_c_double, &
            0.373000_c_double, &
            20.046000_c_double, &
            8.030000_c_double, &
            24.641000_c_double, &
            21.214000_c_double, &
            17.193000_c_double, &
            12.701000_c_double, &
            11.994000_c_double, &
            11.172000_c_double, &
            10.398000_c_double, &
            9.727000_c_double, &
            9.170000_c_double, &
            8.722000_c_double, &
            8.341000_c_double, &
            7.951000_c_double, &
            7.471000_c_double, &
            6.754000_c_double, &
            5.685000_c_double, &
            4.655000_c_double, &
            3.693000_c_double, &
            2.846000_c_double, &
            2.195000_c_double, &
            1.795000_c_double, &
            1.604000_c_double, &
            1.214000_c_double, &
            0.328000_c_double, &
            19.365000_c_double, &
            7.774000_c_double, &
            23.947000_c_double, &
            20.523000_c_double, &
            16.551000_c_double, &
            12.194000_c_double, &
            11.468000_c_double, &
            10.671000_c_double, &
            9.964000_c_double, &
            9.363000_c_double, &
            8.869000_c_double, &
            8.467000_c_double, &
            8.103000_c_double, &
            7.698000_c_double, &
            7.176000_c_double, &
            6.412000_c_double, &
            5.363000_c_double, &
            4.363000_c_double, &
            3.445000_c_double, &
            2.658000_c_double, &
            2.077000_c_double, &
            1.739000_c_double, &
            1.559000_c_double, &
            1.136000_c_double, &
            0.287000_c_double, &
            18.696000_c_double, &
            7.513000_c_double, &
            23.288000_c_double, &
            19.832000_c_double, &
            15.920000_c_double, &
            11.705000_c_double, &
            10.977000_c_double, &
            10.216000_c_double, &
            9.576000_c_double, &
            9.039000_c_double, &
            8.600000_c_double, &
            8.231000_c_double, &
            7.873000_c_double, &
            7.446000_c_double, &
            6.881000_c_double, &
            6.079000_c_double, &
            5.054000_c_double, &
            4.089000_c_double, &
            3.219000_c_double, &
            2.494000_c_double, &
            1.979000_c_double, &
            1.692000_c_double, &
            1.513000_c_double, &
            1.060000_c_double, &
            0.251000_c_double, &
            18.040000_c_double, &
            7.251000_c_double, &
            22.656000_c_double, &
            19.146000_c_double, &
            15.304000_c_double, &
            11.240000_c_double, &
            10.521000_c_double, &
            9.807000_c_double, &
            9.231000_c_double, &
            8.752000_c_double, &
            8.357000_c_double, &
            8.011000_c_double, &
            7.648000_c_double, &
            7.194000_c_double, &
            6.588000_c_double, &
            5.758000_c_double, &
            4.761000_c_double, &
            3.834000_c_double, &
            3.014000_c_double, &
            2.351000_c_double, &
            1.897000_c_double, &
            1.652000_c_double, &
            1.465000_c_double, &
            0.988000_c_double, &
            0.220000_c_double, &
            17.398000_c_double, &
            6.987000_c_double, &
            22.048000_c_double, &
            18.469000_c_double, &
            14.707000_c_double, &
            10.800000_c_double, &
            10.103000_c_double, &
            9.441000_c_double, &
            8.923000_c_double, &
            8.494000_c_double, &
            8.134000_c_double, &
            7.800000_c_double, &
            7.426000_c_double, &
            6.943000_c_double, &
            6.298000_c_double, &
            5.451000_c_double, &
            4.484000_c_double, &
            3.599000_c_double, &
            2.831000_c_double, &
            2.227000_c_double, &
            1.829000_c_double, &
            1.616000_c_double, &
            1.417000_c_double, &
            0.920000_c_double, &
            0.193000_c_double, &
            16.773000_c_double, &
            6.725000_c_double, &
            21.459000_c_double, &
            17.805000_c_double, &
            14.132000_c_double, &
            10.388000_c_double, &
            9.722000_c_double, &
            9.113000_c_double, &
            8.649000_c_double, &
            8.262000_c_double, &
            7.928000_c_double, &
            7.597000_c_double, &
            7.205000_c_double, &
            6.691000_c_double, &
            6.015000_c_double, &
            5.158000_c_double, &
            4.225000_c_double, &
            3.383000_c_double, &
            2.667000_c_double, &
            2.120000_c_double, &
            1.771000_c_double, &
            1.583000_c_double, &
            1.369000_c_double, &
            0.856000_c_double, &
            0.169000_c_double, &
            16.165000_c_double, &
            6.465000_c_double, &
            20.887000_c_double, &
            17.157000_c_double, &
            13.581000_c_double, &
            10.004000_c_double, &
            9.375000_c_double, &
            8.820000_c_double, &
            8.403000_c_double, &
            8.051000_c_double, &
            7.733000_c_double, &
            7.398000_c_double, &
            6.985000_c_double, &
            6.442000_c_double, &
            5.739000_c_double, &
            4.880000_c_double, &
            3.983000_c_double, &
            3.186000_c_double, &
            2.522000_c_double, &
            2.028000_c_double, &
            1.723000_c_double, &
            1.551000_c_double, &
            1.320000_c_double, &
            0.795000_c_double, &
            0.148000_c_double, &
            15.576000_c_double, &
            6.210000_c_double, &
            20.328000_c_double, &
            16.528000_c_double, &
            13.055000_c_double, &
            9.650000_c_double, &
            9.061000_c_double, &
            8.558000_c_double, &
            8.181000_c_double, &
            7.856000_c_double, &
            7.547000_c_double, &
            7.202000_c_double, &
            6.766000_c_double, &
            6.194000_c_double, &
            5.471000_c_double, &
            4.617000_c_double, &
            3.759000_c_double, &
            3.006000_c_double, &
            2.393000_c_double, &
            1.948000_c_double, &
            1.681000_c_double, &
            1.520000_c_double, &
            1.270000_c_double, &
            0.738000_c_double, &
            0.130000_c_double, &
            15.008000_c_double, &
            5.959000_c_double, &
            19.780000_c_double, &
            15.919000_c_double, &
            12.556000_c_double, &
            9.324000_c_double, &
            8.778000_c_double, &
            8.322000_c_double, &
            7.979000_c_double, &
            7.673000_c_double, &
            7.367000_c_double, &
            7.008000_c_double, &
            6.548000_c_double, &
            5.951000_c_double, &
            5.214000_c_double, &
            4.370000_c_double, &
            3.551000_c_double, &
            2.844000_c_double, &
            2.278000_c_double, &
            1.880000_c_double, &
            1.644000_c_double, &
            1.489000_c_double, &
            1.221000_c_double, &
            0.686000_c_double, &
            0.115000_c_double, &
            14.461000_c_double, &
            5.715000_c_double, &
            19.242000_c_double, &
            15.332000_c_double, &
            12.083000_c_double, &
            9.025000_c_double, &
            8.522000_c_double, &
            8.110000_c_double, &
            7.794000_c_double, &
            7.501000_c_double, &
            7.190000_c_double, &
            6.815000_c_double, &
            6.330000_c_double, &
            5.712000_c_double, &
            4.967000_c_double, &
            4.139000_c_double, &
            3.360000_c_double, &
            2.697000_c_double, &
            2.178000_c_double, &
            1.821000_c_double, &
            1.611000_c_double, &
            1.458000_c_double, &
            1.173000_c_double, &
            0.636000_c_double, &
            0.101000_c_double, &
            13.937000_c_double, &
            5.477000_c_double, &
            18.713000_c_double, &
            14.767000_c_double, &
            11.638000_c_double, &
            8.752000_c_double, &
            8.290000_c_double, &
            7.917000_c_double, &
            7.621000_c_double, &
            7.335000_c_double, &
            7.017000_c_double, &
            6.622000_c_double, &
            6.115000_c_double, &
            5.480000_c_double, &
            4.731000_c_double, &
            3.923000_c_double, &
            3.183000_c_double, &
            2.564000_c_double, &
            2.089000_c_double, &
            1.770000_c_double, &
            1.581000_c_double, &
            1.427000_c_double, &
            1.125000_c_double, &
            0.591000_c_double, &
            0.090000_c_double, &
            13.435000_c_double, &
            5.247000_c_double, &
            18.193000_c_double, &
            14.227000_c_double, &
            11.219000_c_double, &
            8.502000_c_double, &
            8.080000_c_double, &
            7.739000_c_double, &
            7.459000_c_double, &
            7.174000_c_double, &
            6.845000_c_double, &
            6.431000_c_double, &
            5.902000_c_double, &
            5.253000_c_double, &
            4.506000_c_double, &
            3.722000_c_double, &
            3.022000_c_double, &
            2.445000_c_double, &
            2.011000_c_double, &
            1.725000_c_double, &
            1.553000_c_double, &
            1.395000_c_double, &
            1.078000_c_double, &
            0.548000_c_double, &
            0.079000_c_double, &
            12.956000_c_double, &
            5.025000_c_double, &
            17.682000_c_double, &
            13.711000_c_double, &
            10.827000_c_double, &
            8.275000_c_double, &
            7.889000_c_double, &
            7.575000_c_double, &
            7.305000_c_double, &
            7.017000_c_double, &
            6.674000_c_double, &
            6.240000_c_double, &
            5.692000_c_double, &
            5.034000_c_double, &
            4.293000_c_double, &
            3.535000_c_double, &
            2.874000_c_double, &
            2.338000_c_double, &
            1.942000_c_double, &
            1.685000_c_double, &
            1.526000_c_double, &
            1.362000_c_double, &
            1.033000_c_double, &
            0.509000_c_double, &
            0.071000_c_double &
    ], shape=[26,25], order=[2,1])

contains

    !> Convert string to lowercase
    !! @param[in] str Input string
    !!
    !! @return lowerStr Lowercase version of input string
    pure function toLower(str) result(lowerStr)
            character(len=*), intent(in) :: str
            character(len=len(str)) :: lowerStr
            integer :: i, ic
            lowerStr = str
            do i = 1, len(str)
                    ic = iachar(str(i:i))
                    if (ic >= 65 .and. ic <= 90) lowerStr(i:i) = achar(ic + 32)
            end do
    end function toLower

    !> Returns the atomic form factor for a given element at a specific Q
    !>
    !! Returns the atomic form factor for a given element at a specific Q.
    !!
    !! @param[in] q Scattering vector magnitude (sin θ)/λ in Å⁻¹
    !! @param[in] element Element symbol (case-insensitive, e.g., 'Fe', 'cu')
    !!
    !! @return f0Val Atomic form factor (electrons)
    function getF0(q, element) result(f0Val)
            real(c_double), intent(in) :: q
            character(len=*), intent(in) :: element
            real(c_double) :: f0Val
            integer :: qIdx, elemIdx, i
            real(c_double) :: qRound
            character(len=10) :: elementLower

            ! Convert element to lower case
            elementLower = toLower(element)

            ! Round q to nearest 0.01 Å⁻¹
            qRound = ceiling(q * 100_c_double) / 100_c_double

            ! Find element index
            elemIdx = -1
            do i = 1, nElements
                    if (trim(elements(i)) == trim(elementLower)) then
                            elemIdx = i
                            EXIT
                    end if
            end do

            if (elemIdx == -1) then
                    write(*, '(A, A, A)') 'ERROR in getF0: Element "', &
                            trim(element), '" not found'
                    error stop
            end if

            ! Find Q index
            qIdx = -1
            do i = 1, nQValues
                    if (abs(qValues(i) - qRound) < 1.0e-6_c_double) then
                            qIdx = i
                            EXIT
                    end if
            end do

            if (qIdx == -1) then
                    write(*, '(A, F12.6, A)') 'ERROR in getF0: Q value ', &
                            qRound, ' not found in table'
                    error stop
            end if

            ! Access f0 value and return
            f0Val = f0Data(qIdx, elemIdx)

    end function getF0

    !> Returns a list of the available q values
    !>
    !! @return qVals List of available q values
    function getQVals() result(qVals)
            real(c_double), allocatable :: qVals(:)
            allocate(qVals, source=qValues)
    end function getQVals
end module F0Factor
