# HJL's DMRG C++ Code


------------

- Contact : 
 _Hyeong Jun Lee_ [hjuntaf (at) gmail.com]

------------


This is a density-matrix renormalization group (DMRG) calculation code within C++.
This program can simulate one-dimensional systems and now primarily supports quantum spin cases.
It calculates approximated thermodynamic ground states with truncations of irrelevant Hilbert space part. 

- Required library: _GSL_
- Parallel compuation is supported with both _MPI_ and _OpenMP_.


Usage
-----

After compiling the source codes, you can run with arguments
```
$ ./<executable> <1.m_keep> <2.final_length-1> <3.Lanczos_precision> <4.Job_info_File> <5.Jend value> <6.Jz value> <7.J2 value> <8.J1 value> -r <10.input_dir>
```
or
```
$ ./<executable> <1.m_keep> <2.final_length-1> <3.Lanczos_precision> <4.Job_info_File> <5.Jend value> <6.Jz value> <7.J2 value> <8.J1n value>
```

References
----------
1.  S. R. White, Phys. Rev. Lett. 69, 2863 (1992).
2.  S. R. White, Phys. Rev. B 48, 10345 (1993).
3.  U. Schollwöck, Rev. Mod. Phys. 77, 259 (2005).
4.  F. Verstraete, V. Murg, and J. I. Cirac, Adv. Phys. 57, 143 (2008).
5.  U. Schollwöck, Ann. Phys. 326, 96 (2011).


