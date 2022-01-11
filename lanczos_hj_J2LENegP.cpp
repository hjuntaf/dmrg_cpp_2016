#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

#ifndef LANCZOS_HJ_J2LENEG_CPP_
#define LANCZOS_HJ_J2LENEG_CPP_

int opfSopModLanczosE( gsl_vector *q0 , double &gggsEval, sparseMatOpform H , double &overlap, size_t &iter, size_t dim ) {
	size_t dimL = 2 ;
	gsl_vector *q1   = gsl_vector_calloc( dim ) ,
		   *qDum = gsl_vector_calloc( dim ) ,
		   *evalL = gsl_vector_alloc( dimL ) ;
	gsl_matrix *evecL = gsl_matrix_alloc( dimL, dimL ) ;

	double *a = new double [dimL*dimL] ;
	gsl_matrix_view aView = gsl_matrix_view_array( a, dimL, dimL ) ;

	//std::cout << "iter:" << iter << "[H.sizeDat,H.size,q0.size,q1.size]=" << H.sizeDat << ','<< H.dim << ',' << q0->size << ',' << q1->size << std::endl ;
	H.opfDotGslJ2Neg( q0 , q1 ) ;	// SET q1 = H*q0
	//std::cout << "dotGsliter:" << iter << std::endl ;
	gsl_blas_ddot( q0, q1,   &(a[0]) ) ; 	// a00
	gsl_blas_daxpy( -a[0] , q0, q1 ) ;	// SET q1 = A*q0 - a*q0 ( q1 = q1 - a*q0 )
	gsl_vector_scale( q1, 1./gsl_blas_dnrm2(q1) ) ;

	H.opfDotGslJ2Neg( q1 , qDum ) ;	// SET H*q1
	gsl_blas_ddot( q0, qDum, &(a[1]) ) ; 	// a01
	a[2] = a[1] ;				// a10
	gsl_blas_ddot( q1, qDum, &(a[3]) ) ; 	// a11

	gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc( 4 ) ;
	gsl_eigen_symmv( &aView.matrix, evalL, evecL, w ) ; 
	gsl_eigen_symmv_sort( evalL, evecL, GSL_EIGEN_SORT_VAL_ASC ) ;
	gsl_eigen_symmv_free( w ) ;

	gsl_vector_memcpy( qDum, q0 ) ;
	gsl_vector_scale( q0, gsl_matrix_get( evecL, 0,0 ) ) ;
	gsl_blas_daxpy( gsl_matrix_get(evecL,1,0) , q1, q0 ) ;	// SET new q0 = alpha*q0 + beta*q1 : ground state in Krylov space.
	gsl_vector_scale( q0, 1./gsl_blas_dnrm2(q0) ) ;

	double dumOverlap=0 ;
	gsl_blas_ddot( q0, qDum,  &dumOverlap ) ; 	// (new q0) dot q0 = overlap

	//have already a newly obtained q0
	gggsEval = gsl_vector_get( evalL , 0 ) ;
	overlap = dumOverlap ; 

	gsl_vector_free( q1 ) ;
	gsl_vector_free( qDum ) ;
	gsl_vector_free( evalL ) ;
	gsl_matrix_free( evecL ) ;
	free( a ) ;

	iter++ ;
	return 0 ;
	delete []a ;
}

int opfSopIterModLanczosRNlpPrev ( sparseMatOpform originH, gsl_vector *initSt, double &ggsEval, size_t matDim , size_t &llanczosIter , double &ddeltaE , char* oooufdir , size_t nnum_site , double ggsEvalPrev , double &nnDecDigit ) {
	double overlap=0 ,
	       dE =0 ;
	size_t i(0) ;

	i=0 ;

	std::stringstream ssOutfIter ;
	ssOutfIter << "mkdir -p " << oooufdir << "/L" << nnum_site ;
	system( ssOutfIter.str().c_str() ) ;
	ssOutfIter.str("") ;
	ssOutfIter << oooufdir << "/L" << nnum_site << "/iter.dat" ;
	ggsEval = 0 ;

	std::ofstream fw( ssOutfIter.str().c_str() , std::ios::app ) ;
	fw << std::scientific ;
	Ldata Kept( ggsEval, initSt, matDim ) ;
	do {
		dE = -ggsEval ;
		opfSopModLanczosE( initSt , ggsEval, originH , overlap, i , matDim ) ;
		dE += ggsEval ;
		Kept.comp( ggsEval, initSt ) ;

		fw << std::setprecision(2) ;
		fw << nnum_site << "\t" << i << "\t" ;
		fw << std::setprecision(16) ;
		fw << ggsEval/nnum_site/2. << "\t" << overlap << "\t" << dE << "\t"<< (ggsEval-ggsEvalPrev)/2. << std::endl ;
	} while ( Kept.def < 10 ) ;
	fw.close() ;
	Kept.ret( ggsEval, initSt ) ;
	nnDecDigit = log10( 1-overlap ) ;
	ddeltaE = - dE ;
	llanczosIter = i ;
	return 0 ;
}

int opfSopModLanczos ( gsl_vector *q0 , double &gggsEval, sparseMatOpform H , double &overlap, size_t &iter, size_t dim ) {
	size_t dimL = 2 ;
	gsl_vector *q1   = gsl_vector_calloc( dim ) ,
		   *qDum = gsl_vector_calloc( dim ) ,
		   *evalL = gsl_vector_alloc( dimL ) ;
	gsl_matrix *evecL = gsl_matrix_alloc( dimL, dimL ) ;

	double *a = new double [dimL*dimL] ;
	gsl_matrix_view aView = gsl_matrix_view_array( a, dimL, dimL ) ;

	//std::cout << "iter:" << iter << "[H.sizeDat,H.size,q0.size,q1.size]=" << H.sizeDat << ','<< H.dim << ',' << q0->size << ',' << q1->size << std::endl ;
	H.opfDotGslJ2( q0 , q1 ) ;	// SET q1 = H*q0
	//std::cout << "dotGsliter:" << iter << std::endl ;
	gsl_blas_ddot( q0, q1,   &(a[0]) ) ; 	// a00
	gsl_blas_daxpy( -a[0] , q0, q1 ) ;	// SET q1 = A*q0 - a*q0 ( q1 = q1 - a*q0 )
	gsl_vector_scale( q1, 1./gsl_blas_dnrm2(q1) ) ;

	H.opfDotGslJ2( q1 , qDum ) ;	// SET H*q1
	gsl_blas_ddot( q0, qDum, &(a[1]) ) ; 	// a01
	a[2] = a[1] ;				// a10
	gsl_blas_ddot( q1, qDum, &(a[3]) ) ; 	// a11

	gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc( 4 ) ;
	gsl_eigen_symmv( &aView.matrix, evalL, evecL, w ) ; 
	gsl_eigen_symmv_sort( evalL, evecL, GSL_EIGEN_SORT_VAL_ASC ) ;
	gsl_eigen_symmv_free( w ) ;

	gggsEval = gsl_vector_get( evalL , 0 ) ;
	gsl_vector_memcpy( qDum, q0 ) ;
	gsl_vector_scale( q0, gsl_matrix_get( evecL, 0,0 ) ) ;
	gsl_blas_daxpy( gsl_matrix_get(evecL,1,0) , q1, q0 ) ;	// SET new q0 = alpha*q0 + beta*q1 : ground state in Krylov space.
	gsl_vector_scale( q0, 1./gsl_blas_dnrm2(q0) ) ;

	gsl_blas_ddot( q0, qDum,  &overlap ) ; 	// (new q0) dot q0 = overlap

	gsl_vector_free( q1 ) ;
	gsl_vector_free( qDum ) ;
	gsl_vector_free( evalL ) ;
	gsl_matrix_free( evecL ) ;
	free( a ) ;

	iter++ ;
	return 0 ;
	delete []a ;
}
int opfSopIterModLanczosRNlp ( sparseMatOpform originH, gsl_vector *initSt, double &ggsEval, size_t matDim , size_t &llanczosIter , double &ddeltaE , char* oooufdir , size_t nnum_site , double ggsEvalPrev , double &nnDecDigit ) {
	double overlap ,
	       dOverlap1=0 ,
	       dOverlap2=9 ;
	size_t i(0) ;

	srand(1) ;
	for( i=0 ; i<matDim ; i++ ) 
		gsl_vector_set( initSt , i , rand()/double(RAND_MAX) - 0.5 ) ;
	gsl_vector_scale( initSt, 1./gsl_blas_dnrm2(initSt) ) ;

	i=0 ;

	ggsEval = 0 ;
	std::stringstream ssOutfIter ;
	ssOutfIter << "mkdir -p " << oooufdir << "/L" << nnum_site ;
	system( ssOutfIter.str().c_str() ) ;
	ssOutfIter.str("") ;
	ssOutfIter << oooufdir << "/L" << nnum_site << "/iter.dat" ;
	do {
		dOverlap1 = dOverlap2 ;
		dOverlap2 = -overlap ;
		ddeltaE = -ggsEval ;
		opfSopModLanczos( initSt , ggsEval, originH , overlap, i , matDim ) ;
		ddeltaE += ggsEval ;
		dOverlap2 += overlap ;

		std::ofstream fw( ssOutfIter.str().c_str() , std::ios::app ) ;
		fw << std::setprecision(2) << std::setw(5) ;
		fw << nnum_site << "\t" << i << "\t" ;
		fw << std::setprecision(16) << std::setw(19) << std::fixed ;
		fw << ggsEval/nnum_site/2. << "\t" << overlap << "\t" << ddeltaE << "\t"<< (ggsEval-ggsEvalPrev)/2. << std::endl ;
		fw.close() ;
	} while( dOverlap2 > -100*fabs(dOverlap1) ) ;
	nnDecDigit = log10( 1-overlap ) ;
	llanczosIter = i ;
	return 0 ;
}
#endif
