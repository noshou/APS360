(* F0Parser.ml *)

exception Invalid_csv  of string

(* loads q, e, f into lists from csv file. *)
let load_data fp = 
	
	(* list of form factors representing rows *)
	let q = ref [] in 

	(* list of elements representing columns *)
	let e = ref [] in 

	(* 2x2 list where each element f[q][i] is a form factor *)
	let f = ref [] in 

	(Csv.load fp) 
	|> List.iteri (

        (* iterate through rows; first row is header, add to list of elements *)
        fun idx row ->

			(* skip first column, add element names to list *)
			if idx == 0 then  begin 
                let rec add_to (list : string list) (accum : string list ) = match list with 
                    | []            -> accum
                    | elm :: list  -> add_to list (elm :: accum) 
                in e := add_to (List.tl row) [];
			end 
			
			(* for n rows: first column goes to q, rest go to f *)
			else match row with 
				| qVal :: f_vals ->

					(* round to nearest 100th *)
					let q_rnd = (Float.round ((float_of_string qVal) *. 100.)) /. 100. in
					q := q_rnd :: !q;

                let rec add_to (list : string list) (accum : float list ) = match list with 
                    | []           -> accum
                    | elm :: list  -> add_to list ((float_of_string elm) :: accum) 
                in f := (add_to f_vals []) :: !f
				
              | [] -> ();
		);

	(* return lists *)
	(List.rev !q, !e, List.rev !f)


(* writes parsed data to Fortran *)
let f0_toFortran csv_fp = 

	(* fill q, e, f with data *)
	let q,e,f = load_data csv_fp in 

	(* write to Fortran at specified output *)
	let oc = open_out "F0Factor.f90" in 
	
	(* Write module header with documentation *)
	Printf.fprintf oc "!> Atomic form factors f0 for X-ray scattering calculations\n";
	Printf.fprintf oc "!>\n";
	Printf.fprintf oc "!! Atomic form factors f0 for X-ray scattering calculations.\n";
	Printf.fprintf oc "!! f0(Q) represents the scattering amplitude of an atom as a function\n";
	Printf.fprintf oc "!! of the scattering vector Q = (sin θ)/λ.\n";
	Printf.fprintf oc "!!\n";
	Printf.fprintf oc "!! Data Source:\n";
	Printf.fprintf oc "!!   International Tables for Crystallography Vol. C\n";
	Printf.fprintf oc "!!   DOI: 10.1107/97809553602060000600\n";
	Printf.fprintf oc "!!\n";
	Printf.fprintf oc "module F0Factor\n";
	Printf.fprintf oc "    use iso_c_binding, only: c_double\n";
	Printf.fprintf oc "    implicit none\n\n";
	Printf.fprintf oc "    private\n\n";
	
	Printf.fprintf oc "    ! Public interface\n";
	Printf.fprintf oc "    public :: nElements, nQValues\n";
	Printf.fprintf oc "    public :: getF0, getQVals\n\n";
	
	Printf.fprintf oc "    !---------------------------------------------------------------------------\n";
	Printf.fprintf oc "    ! Module Data\n";
	Printf.fprintf oc "    !---------------------------------------------------------------------------\n\n";
	
	Printf.fprintf oc "    !> Number of elements in lookup table\n";
	Printf.fprintf oc "    integer, parameter :: nElements = %d\n" (List.length e); 
	Printf.fprintf oc "    !> Number of Q values in lookup table\n";
	Printf.fprintf oc "    integer, parameter :: nQValues = %d\n\n" (List.length q);
	
	(* write element names to array *)
	Printf.fprintf oc "    !> Element/ion symbols (lowercase)\n";
	Printf.fprintf oc "    character(len=4), parameter :: elements(%d) = [ &\n" (List.length e);
	List.iteri (
		fun idx element -> 
        if (idx == 0) then 
            Printf.fprintf oc "            '%-4s'" element 
        else 
            Printf.fprintf oc ", &\n            '%-4s'" element 
		) e;
    Printf.fprintf oc " ]\n\n";
    
	(* write q values to array *)
	Printf.fprintf oc "    !> Scattering vector magnitudes: Q = (sin θ)/λ in Å⁻¹\n";
	Printf.fprintf oc "    !! Range: 0 to ~2.0 Å⁻¹ in increments of 0.01 Å⁻¹\n";
	Printf.fprintf oc "    real(c_double), parameter :: qValues(%d) = [ &\n" (List.length q);
	List.iteri (
    fun idx q -> 
        if (idx == 0) then 
            Printf.fprintf oc "            %f_c_double" q 
        else
            Printf.fprintf oc ", &\n            %f_c_double" q
	) q;
	Printf.fprintf oc " ]\n\n";

	(* write f0Data*)
	Printf.fprintf oc "    !> Form factor data: f0(Q) for each element\n";
	Printf.fprintf oc "    !! Rows: Q values, Columns: Elements\n";
	Printf.fprintf oc "    !! f0 decreases with increasing Q due to destructive interference\n";
	Printf.fprintf oc "    real(c_double), parameter :: f0Data(%d,%d) = reshape([ &\n" (List.length q) (List.length e);
	List.iter (
    let idx = ref 0 in 
    fun row -> List.iter (
        fun col -> 
            if !idx == 0 then Printf.fprintf oc "            %f_c_double" col 
            else Printf.fprintf oc ", &\n            %f_c_double" col;
        idx := !idx + 1;
    ) row  
	) f;
	Printf.fprintf oc " &\n    ], shape=[%d,%d], order=[2,1])\n\n" (List.length q) (List.length e);
	
	(* methods *)
	Printf.fprintf oc "contains\n\n";

  Printf.fprintf oc "    !> Convert string to lowercase\n";
  Printf.fprintf oc "    !! @param[in] str Input string\n";
  Printf.fprintf oc "    !!\n";
  Printf.fprintf oc "    !! @return lowerStr Lowercase version of input string\n";
  Printf.fprintf oc "    pure function toLower(str) result(lowerStr)\n";
  Printf.fprintf oc "            character(len=*), intent(in) :: str\n";
  Printf.fprintf oc "            character(len=len(str)) :: lowerStr\n";
  Printf.fprintf oc "            integer :: i, ic\n";
  Printf.fprintf oc "            lowerStr = str\n";
  Printf.fprintf oc "            do i = 1, len(str)\n";
  Printf.fprintf oc "                    ic = iachar(str(i:i))\n";
  Printf.fprintf oc "                    if (ic >= 65 .and. ic <= 90) lowerStr(i:i) = achar(ic + 32)\n";
  Printf.fprintf oc "            end do\n";
  Printf.fprintf oc "    end function toLower\n\n";


	(* initialize f0 data *)
	Printf.fprintf oc "    !> Returns the atomic form factor for a given element at a specific Q\n";
	Printf.fprintf oc "    !>\n";
	Printf.fprintf oc "    !! Returns the atomic form factor for a given element at a specific Q.\n";
	Printf.fprintf oc "    !!\n";
	Printf.fprintf oc "    !! @param[in] q Scattering vector magnitude (sin θ)/λ in Å⁻¹\n";
	Printf.fprintf oc "    !! @param[in] element Element symbol (case-insensitive, e.g., 'Fe', 'cu')\n";
	Printf.fprintf oc "    !!\n";
	Printf.fprintf oc "    !! @return f0Val Atomic form factor (electrons)\n";
	Printf.fprintf oc "    function getF0(q, element) result(f0Val)\n";
	Printf.fprintf oc "            real(c_double), intent(in) :: q\n";
	Printf.fprintf oc "            character(len=*), intent(in) :: element\n";
	Printf.fprintf oc "            real(c_double) :: f0Val\n";
	Printf.fprintf oc "            integer :: qIdx, elemIdx, i\n";
	Printf.fprintf oc "            real(c_double) :: qRound\n";
	Printf.fprintf oc "            character(len=10) :: elementLower\n\n";
	Printf.fprintf oc "            ! Convert element to lower case\n";
	Printf.fprintf oc "            elementLower = toLower(element)\n\n";
	Printf.fprintf oc "            ! Round q to nearest 0.01 Å⁻¹\n";
	Printf.fprintf oc "            qRound = ceiling(q * 100_c_double) / 100_c_double\n\n";
	Printf.fprintf oc "            ! Find element index\n";
	Printf.fprintf oc "            elemIdx = -1\n";
	Printf.fprintf oc "            do i = 1, nElements\n";
	Printf.fprintf oc "                    if (trim(elements(i)) == trim(elementLower)) then\n";
	Printf.fprintf oc "                            elemIdx = i\n";
	Printf.fprintf oc "                            EXIT\n";
	Printf.fprintf oc "                    end if\n";
	Printf.fprintf oc "            end do\n\n";
	Printf.fprintf oc "            if (elemIdx == -1) then\n";
	Printf.fprintf oc "                    write(*, '(A, A, A)') 'ERROR in getF0: Element \"', &\n";
	Printf.fprintf oc "                            trim(element), '\" not found'\n";
	Printf.fprintf oc "                    error stop\n";
	Printf.fprintf oc "            end if\n\n";
	Printf.fprintf oc "            ! Find Q index\n";
	Printf.fprintf oc "            qIdx = -1\n";
	Printf.fprintf oc "            do i = 1, nQValues\n";
	Printf.fprintf oc "                    if (abs(qValues(i) - qRound) < 1.0e-6_c_double) then\n";
	Printf.fprintf oc "                            qIdx = i\n";
	Printf.fprintf oc "                            EXIT\n";
	Printf.fprintf oc "                    end if\n";
	Printf.fprintf oc "            end do\n\n";
	Printf.fprintf oc "            if (qIdx == -1) then\n";
	Printf.fprintf oc "                    write(*, '(A, F12.6, A)') 'ERROR in getF0: Q value ', &\n";
	Printf.fprintf oc "                            qRound, ' not found in table'\n";
	Printf.fprintf oc "                    error stop\n";
	Printf.fprintf oc "            end if\n\n";
	Printf.fprintf oc "            ! Access f0 value and return\n";
	Printf.fprintf oc "            f0Val = f0Data(qIdx, elemIdx)\n\n";
	Printf.fprintf oc "    end function getF0\n\n";
	Printf.fprintf oc "    !> Returns a list of the available q values\n";
	Printf.fprintf oc "    !>\n";
	Printf.fprintf oc "    !! @return qVals List of available q values\n";
	Printf.fprintf oc "    function getQVals() result(qVals)\n";
	Printf.fprintf oc "            real(c_double), allocatable :: qVals(:)\n";
  Printf.fprintf oc "            allocate(qVals, source=qValues)\n";
  Printf.fprintf oc "    end function getQVals\n";
  Printf.fprintf oc "end module F0Factor\n";
	close_out oc

(* Main entry point *)
let () =
  if Array.length Sys.argv != 2 then begin
    Printf.eprintf "Usage: %s <input.csv>\n" Sys.argv.(0);
    Printf.eprintf "  input.csv - CSV file with f0 form factor data\n";
    Printf.eprintf "  Outputs to: F0Factor.f90\n";
    exit 1
  end;
  
  try
    f0_toFortran Sys.argv.(1)
  with 
  | Sys_error msg -> 
    Printf.eprintf "Error: %s\n" msg;
    exit 1
  | Csv.Failure (row, col, msg) ->
    Printf.eprintf "CSV Error at row %d, column %d: %s\n" row col msg;
    exit 1