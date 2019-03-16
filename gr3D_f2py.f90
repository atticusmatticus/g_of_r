! calculate force from each solvent atom to solute atom using f2py fortran subroutine for python. To create the executable for this
! subroutine enter the following command: f2py -c -m gr3D_f2py gr3D_f2py.f90
! This will create an executable with the name of the first argument 'gr3D_f2py' followed by the '.so' extension using the
! 'gr3D_f2py.f90' file, (this file), to create it.

! module of functions to prevent compilation errors
module my_subs
	implicit none
contains
	! calculate cross product of two vectors, 'a' and 'b'
	function cross_product(a, b)
		implicit none
		real(kind=8), dimension(3)	:: cross_product	! output
		real(kind=8), intent(in)	:: a(3), b(3)		! inputs not to be changed

		cross_product(1) = a(2)*b(3) - a(3)*b(2)
		cross_product(2) = a(3)*b(1) - a(1)*b(3)
		cross_product(3) = a(1)*b(2) - a(2)*b(1)
	end function cross_product
end module my_subs


! calculate the force from each atom of CL3 and project along r, s, and t.
subroutine calc_force( solv_atoms, n_atoms, atom_type_index, n_types, nb_parm_index, solute_index, solute_position, &
		& solvent_position, box, hbox, lj_a_coeff, lj_b_coeff, ljdr, r_vec, force_var )
	use my_subs
	implicit none
	! Note: f2py variables from python
	integer			:: solv_atoms, n_atoms, n_types, solute_index, atom_type_index(n_atoms), nb_parm_index(n_types*n_types)
	real(kind=8)	:: solute_position(3), box(3), hbox(3), ljdr(3), solvent_position(solv_atoms,3), &
		& lj_a_coeff(n_types*(n_types+1)/2), lj_b_coeff(n_types*(n_types+1)/2), r_vec(3), force_var(3)

	! Note: f2py lines for input and output variables from python script

!f2py intent(in) solv_atoms, n_atoms, atom_type_index, n_types, nb_parm_index, solute_index, solute_position, solvent_position, box
!f2py intent(in) hbox, lj_a_coeff, lj_b_coeff, ljdr, r_vec
!f2py intent(out) force_var

	! Note: variables exclusively in this code
	integer			:: i, j, amber_index, nb_index
	real(kind=8)	:: solvAtom_force_vec(3), r6, solvAtom_dist2, solvAtom_dr(3), p_vec(3), s_vec(3), t_vec(3)

	solvAtom_force_vec = 0.0d0
	do i = 1, solv_atoms
		amber_index = n_types * (atom_type_index(solute_index+1) - 1) + atom_type_index(i+2)
		nb_index = nb_parm_index(amber_index)
		call computePbcDist2(solute_position, solvent_position, i, solv_atoms, box, hbox, solvAtom_dist2, solvAtom_dr)
		r6 = solvAtom_dist2**(-3)
		do j = 1, 3
			solvAtom_force_vec(j) = solvAtom_force_vec(j) + (r6 * (12 * r6 * lj_a_coeff(nb_index) - 6 * lj_b_coeff(nb_index)) / &
				& solvAtom_dist2) * solvAtom_dr(j)
		end do
	end do

	! project the total force onto r, s, and t vectors.
	p_vec = solvent_position(1,:) - solvent_position(2,:) ! points toward H from C
	t_vec = cross_product(r_vec, p_vec) ! set t_vec to the cross_product of r and p
	t_vec = t_vec / norm2(t_vec)
	s_vec = cross_product(t_vec, r_vec) ! set s_vec to the cross_product of t and r
	s_vec = s_vec / norm2(s_vec)

	force_var(1) = dot_product(dot_product(solvAtom_force_vec, r_vec)*r_vec, ljdr)
	force_var(2) = dot_product(dot_product(solvAtom_force_vec, s_vec)*s_vec, ljdr)
	force_var(3) = dot_product(dot_product(solvAtom_force_vec, t_vec)*t_vec, ljdr)

end subroutine calc_force


! gets called in calc_force
! calculate displacement vector and distance squared of that vector with periodic boundary conditions.
subroutine computePbcDist2(r1, r2, i, solv_atoms, box, hbox, dist2, dr)
	implicit none
	integer			:: i, solv_atoms, j
	real(kind=8)	:: r1(3), r2(solv_atoms,3), box(3), hbox(3), dist2, dr(3)
	
	do j = 1, 3
		dr(j) = r1(j) - r2(i,j) ! points from r2 to r1
	end do
	do j = 1, 3
		if (dr(j) .lt. -hbox(j)) then
			dr(j) = dr(j) + box(j)
		else if (dr(j) .gt. hbox(j)) then
			dr(j) = dr(j) - box(j)
		end if
		dist2 = dot_product(dr,dr)
	end do

end subroutine computePbcDist2
