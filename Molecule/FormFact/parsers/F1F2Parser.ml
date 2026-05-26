(* F1F2Parser.ml *)

open Yojson.Basic

exception Invalid_element of string 

(* regex to extract element symbol *)
let reElm = Str.regexp "\\([A-Za-z]+\\)\\([0-9]*\\)\\([0-9]*[+-]\\)?"

(* f1 and f2 values *)
let getF1F2 (elm : string) lookup : float * float = 
  let elmLower = 
    if (Str.string_match reElm elm 0) then
      Str.matched_group 1 elm
    else
      raise (Invalid_element elm)
  in 
  try
    let lst = 
      Util.to_list (List.nth (Util.to_list (lookup |> Util.member elmLower)) 0) 
    in
    (Util.to_float (List.nth lst 0), Util.to_float (List.nth lst 1))
  with _ ->
    raise (Invalid_element elm) 

(* Generate Fortran module from JSON data *)
let generate_fortran_module jsonFp =
  let oc = open_out "F1F2Factors.f90" in
  
  let lookup = from_file jsonFp in

  (* Write module header with documentation *)
  Printf.fprintf oc "!> Anomalous scattering factors f1 and f2 for X-ray diffraction\n";
  Printf.fprintf oc "!>\n";
  Printf.fprintf oc "!! Provides anomalous scattering factors f1 and f2 for X-ray diffraction.\n";
  Printf.fprintf oc "!!\n";
  Printf.fprintf oc "!! Anomalous scattering factors (f' and f'', or f1 and f2) account for\n";
  Printf.fprintf oc "!! energy-dependent corrections to atomic scattering factors near absorption edges.\n";
  Printf.fprintf oc "module F1F2Factors\n";
  Printf.fprintf oc "    implicit none\n";
  Printf.fprintf oc "    private\n\n";
  
  Printf.fprintf oc "    ! Public interface\n";
  Printf.fprintf oc "    public :: getF1F2\n\n";
  
  (* Extract all element keys *)
  let elements = match lookup with
    | `Assoc assoc -> List.map fst assoc
    | _ -> []
  in
  
  (* Write number of elements *)
  Printf.fprintf oc "    !---------------------------------------------------------------------------\n";
  Printf.fprintf oc "    ! Module Data\n";
  Printf.fprintf oc "    !---------------------------------------------------------------------------\n\n";
  Printf.fprintf oc "    !> Number of elements in lookup table\n";
  Printf.fprintf oc "    integer, parameter :: nElements = %d\n\n" (List.length elements);
  
  (* Write element names array *)
  Printf.fprintf oc "    !> Element symbols (lowercase)\n";
  Printf.fprintf oc "    character(len=2), dimension(nElements), parameter :: elements = [ &\n";
  List.iteri (fun i elm ->
    let padding = String.make (max 0 (2 - String.length elm)) ' ' in
    Printf.fprintf oc "        '%s%s'%s\n" elm padding 
      (if i < List.length elements - 1 then ", &" else " ]")
  ) elements;
  Printf.fprintf oc "\n";
  
  (* Write f1 values array *)
  Printf.fprintf oc "    !> f1 values (real part of anomalous scattering factor)\n";
  Printf.fprintf oc "    !! Also known as f' or Δf'\n";
  Printf.fprintf oc "    real(8), dimension(nElements), parameter :: f1Values = [ &\n";
  List.iteri (fun i elm ->
    let (f1, _) = getF1F2 elm lookup in
    Printf.fprintf oc "        %e%s\n" f1
      (if i < List.length elements - 1 then ", &" else " ]")
  ) elements;
  Printf.fprintf oc "\n";
  
  (* Write f2 values array *)
  Printf.fprintf oc "    !> f2 values (imaginary part of anomalous scattering factor)\n";
  Printf.fprintf oc "    !! Also known as f'' or Δf''\n";
  Printf.fprintf oc "    real(8), dimension(nElements), parameter :: f2Values = [ &\n";
  List.iteri (fun i elm ->
    let (_, f2) = getF1F2 elm lookup in
    Printf.fprintf oc "        %e%s\n" f2
      (if i < List.length elements - 1 then ", &" else " ]")
  ) elements;
  Printf.fprintf oc "\n";
  
  (* Write contains section *)
  Printf.fprintf oc "contains\n\n";
  
  (* Write lookup function *)
  Printf.fprintf oc "    !> Retrieve anomalous scattering factors f1 and f2 for a given element\n";
  Printf.fprintf oc "    !>\n";
  Printf.fprintf oc "    !! Retrieves the anomalous scattering factors f1 and f2 for a given energy level.\n";
  Printf.fprintf oc "    !! Element symbol is case-insensitive and numeric suffixes are ignored.\n";
  Printf.fprintf oc "    !!\n";
  Printf.fprintf oc "    !! @param[in]  elm    Element symbol (e.g., 'Fe', 'Cu', 'Zn')\n";
  Printf.fprintf oc "    !! @param[out] f1     Real part of anomalous scattering factor (f')\n";
  Printf.fprintf oc "    !! @param[out] f2     Imaginary part of anomalous scattering factor (f'')\n";
  Printf.fprintf oc "    !! @param[out] status Return status: 0 = success, -1 = element not found\n";
  Printf.fprintf oc "    !!\n";
  Printf.fprintf oc "    subroutine getF1F2(elm, f1, f2, status)\n";
  Printf.fprintf oc "        character(len=*), intent(in) :: elm\n";
  Printf.fprintf oc "        real(8), intent(out) :: f1, f2\n";
  Printf.fprintf oc "        integer, intent(out) :: status\n";
  Printf.fprintf oc "        character(len=:), allocatable :: elmClean\n";
  Printf.fprintf oc "        integer :: i\n";
  Printf.fprintf oc "        character :: c\n\n";
  
  Printf.fprintf oc "        ! Extract alphabetic part only and convert to lowercase\n";
  Printf.fprintf oc "        elmClean = ''\n";
  Printf.fprintf oc "        do i = 1, len_trim(elm)\n";
  Printf.fprintf oc "            c = elm(i:i)\n";
  Printf.fprintf oc "            if ((c >= 'A' .and. c <= 'Z')) then\n";
  Printf.fprintf oc "                elmClean = elmClean // char(iachar(c) + 32)\n";
  Printf.fprintf oc "            else if (c >= 'a' .and. c <= 'z') then\n";
  Printf.fprintf oc "                elmClean = elmClean // c\n";
  Printf.fprintf oc "            else\n";
  Printf.fprintf oc "                exit  ! Stop at first non-alphabetic character\n";
  Printf.fprintf oc "            end if\n";
  Printf.fprintf oc "        end do\n\n";
  
  Printf.fprintf oc "        ! Search for element in lookup table\n";
  Printf.fprintf oc "        status = -1  ! Not found by default\n";
  Printf.fprintf oc "        do i = 1, nElements\n";
  Printf.fprintf oc "            if (trim(elmClean) == trim(elements(i))) then\n";
  Printf.fprintf oc "                f1 = f1Values(i)\n";
  Printf.fprintf oc "                f2 = f2Values(i)\n";
  Printf.fprintf oc "                status = 0  ! Success\n";
  Printf.fprintf oc "                return\n";
  Printf.fprintf oc "            end if\n";
  Printf.fprintf oc "        end do\n\n";
  
  Printf.fprintf oc "        ! Element not found - f1 and f2 remain uninitialized\n\n";
  
  Printf.fprintf oc "    end subroutine getF1F2\n\n";
  
  Printf.fprintf oc "end module F1F2Factors\n";
  
  close_out oc

(* Main entry point *)
let () =
  if Array.length Sys.argv != 2 then begin
    Printf.eprintf "Usage: %s <input.json>\n" Sys.argv.(0);
    Printf.eprintf "  input.json - JSON file with f1/f2 anomalous scattering factors\n";
    Printf.eprintf "  Outputs to: F1F2Factors.f90\n";
    exit 1
  end;
  
  let jsonFile = Sys.argv.(1) in
  
  try
    generate_fortran_module jsonFile
  with
  | Sys_error msg -> 
      Printf.eprintf "Error: %s\n" msg;
      exit 1
  | Yojson.Json_error msg ->
      Printf.eprintf "JSON Error: %s\n" msg;
      exit 1
  | Invalid_element elm ->
      Printf.eprintf "Error: Invalid element '%s' in JSON data\n" elm;
      exit 1