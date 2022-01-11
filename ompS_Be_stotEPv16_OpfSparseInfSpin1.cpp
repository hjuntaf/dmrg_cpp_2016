#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <vector>
#define DIM 3
#define DIMH 2
#define ZEROCUT 1e-15
#define PI 3.141592653589793238
using namespace std ;
#include "class_sparse_J2NegE_ompS.cpp"
#include "lanczos_hj_J2LENegP.cpp"
#include "class_block_J2NegEP.cpp"


int main( int argc, char ** argv ) {
	if( strcmp( "-r", argv[argc-2] ) == 0 ) {
		if( argc != 11 ) {
			printf("Usage : %s <1.m_keep> <2.final_length-1> <3.Lanczos_precision> <4.Job_info_File> <5.Jend value> <6.Jz value> <7.J2 value> <8.J1 value> -r <10.input_dir>\n", argv[0] );
			exit(1);
		}
	}
	else if( argc != 9 ){
		printf("Usage : %s <1.m_keep> <2.final_length-1> <3.Lanczos_precision> <4.Job_info_File> <5.Jend value> <6.Jz value> <7.J2 value> <8.J1n value>\n", argv[0] );
		exit(1);
	}
	supSubBlockChiS *sys ; // sup_sub_block *sys ;
	size_t maxL = atoi( argv[2] ) ;
	sys = ( supSubBlockChiS * ) malloc( sizeof(supSubBlockChiS) * 2 ) ;
	const sopSpinOne spinSop ;
	const sopSpinHalfCoup spinSopH ;
	const size_t m_keep = atoi( argv[1] )  ;
	const size_t lanczosPSet = atoi( argv[3] ) ;
	const double Jend = atof( argv[5] ) ;
	const double Jz   = atof( argv[6] ) ;
	const double J2   = atof( argv[7] ) ;
	const double J1   = atof( argv[8] ) ;

	char outfname[255] ,
	     outfnamew[255] ,
	     outfnamec[255] ,
	     outfnametouch[255] ,
	     outdir[255] ,
	     outdirw[255] ,
	     outfdir[255] ,
	     outfdirw[255] ,
	     dum[255] ,
	     dum2[255] ;
	size_t  i=0 ;

	double dumVal = 0 ;
	for( i=0 ; i<lanczosPSet ; i++ ) {
		dumVal = dumVal + pow(0.1 , i)*0.9 ;
	}
	char verInfo[255] ;
	char verDir [255] ;

	sprintf( verInfo , "L%zu", maxL ) ;
	sprintf( verDir  , "v16/ompBeStotEOpfMSLanczos") ;
	cout << "VD:"<<verDir <<endl ;
	sprintf( outdir , "/ahome/jun/dmrg/heisenberg/spin1/data/%s", verDir ) ;
	sprintf( outdirw, "/work/jun/dmrg/heisenberg/spin1/data/%s", verDir ) ;
	cout << "OD:"<<outdirw <<endl ;

	sprintf( dum , "test -d %s", outfdir ) ;
	if( system(dum) != 0 ) {
		sprintf( dum2   , "mkdir -p %s", outfdir ) ;
		system(dum2) ;
	}

	i=0 ;
	do{
		sprintf( outfname , "%s/m%zu/Jend%.1f/Jz%.2f/J2%.3f/J1n%g/%s_%zu.dat" , outdir , m_keep ,Jend, Jz , J2 , J1 , verInfo , i ) ;
		sprintf( dum , "test -s %s", outfname ) ;
		i++ ;
	} while( system(dum) == 0 ) ;
	i -= 1 ;
	sprintf( outfnamew,"%s/m%zu/Jend%.1f/Jz%.2f/J2%.3f/J1n%g/%s_%zu.dat" , outdirw, m_keep ,Jend, Jz , J2 , J1 , verInfo , i ) ;
	sprintf( outfdir , "%s/m%zu/Jend%.1f/Jz%.2f/J2%.3f/J1n%g/dir_%s_%zu" , outdir , m_keep ,Jend, Jz , J2 , J1 , verInfo , i ) ;
	sprintf( outfdirw, "%s/m%zu/Jend%.1f/Jz%.2f/J2%.3f/J1n%g/dir_%s_%zu" , outdirw, m_keep ,Jend, Jz , J2 , J1 , verInfo , i ) ;
	sprintf( outfnamec,"%s/m%zu_Jend%.1f_Jz%.2f_J2%.3f_J1n%g_dir_%s_%zu.tar.gz" , outdirw, m_keep ,Jend, Jz , J2 , J1 , verInfo , i ) ;
	sprintf( outfnametouch,"%s/m%zu_Jend%.1f_Jz%.2f_J2%.3f_J1n%g_dir_%s_%zu" , outdir, m_keep ,Jend, Jz , J2 , J1 , verInfo , i ) ;
	sprintf( dum , "mkdir -p %s", outfdir ) ;
	system( dum ) ;
	sprintf( dum , "mkdir -p %s", outfdirw) ;
	system( dum ) ;
	sprintf( dum , "touch %s", outfname ) ;
	system( dum ) ;

	sprintf( dum , "echo 'cat %s >> %s.info' >> %s", argv[4] , outfnametouch , argv[4] ) ;
	system( dum ) ;
	sprintf( outfdirw, "%s/m%zu/Jend%.1f/Jz%.2f/J2%.3f/J1n%g/dir_%s_%zu" , outdirw, m_keep ,Jend, Jz , J2 , J1 , verInfo , i ) ;

	if( strcmp( "-r", argv[argc-2] ) == 0 ) {
		sys[0].loadSparse( argv[argc-1] , spinSop ) ;
		sys[0].setJz ( Jz ) ;
		sys[0].setJ2 ( J2 ) ;
		sys[0].setJ1n( J1 ) ;
	}
	else {

	sys[0].set_init_num( 2, DIM, DIM ) ;
	sys[0].setInitSopChiNegBe( spinSop, Jend , Jz , J2 , J1 ) ;
	sys[0].setInitPop() ;
	sys[0].opfSopLanczosDiagSymSupblock2Neg( m_keep , outfdirw ) ;
	
	// Before activating calc() options, must CHECK the update variables.
	sys[0].calcCPSzWrite	 ( outfdirw ) ;
	//sys[0].calcCPSzCorrWrite ( outfdirw ) ;
	sys[0].calcCPChiCorrWrite( outfdirw ) ;
	sys[0].calcPSxyCorrE     ( outfdirw ) ;
	sys[0].calcPStotCorrE    ( outfdirw ) ;
	sys[0].calcPStot         ( outfdirw ) ;
	sys[0].calcPSxzStrCorrE   ( outfdirw ) ; // string corr
	sys[0].calcPSxzSstrCorrE  ( outfdirw ) ; // string corr
	sys[0].calcEE		 ( outfdirw ) ;
	//sys[0].redEvecWrite 	 ( outfdirw ) ; // optional
	sys[0].write  		 ( outfnamew) ;
	sys[0].saveNDelSparse	 ( outfdirw ) ;

	sys[0].freeInter() ;
	}

	short int ind =0 ;
	short int ind2=0 ;
	size_t L = sys[0].L() ;
	for( i=1 ; L<maxL ; i++ ) {
		ind  = (short int)(i%2) ;
		ind2 = (short int)((i+1)%2) ;
		sys[ind].updateSop2NegS( sys[ind2] , Jz , J2 , J1 ) ;
		sys[ind].opfSopLanczosDiagSymSupblock2Neg( m_keep , outfdirw ,sys[ind2] ) ;

		// Before activating calc() options, must CHECK the update variables.
		sys[ind].calcCPSzWrite	   ( outfdirw ) ;
		//sys[ind].calcCPSzCorrWrite ( outfdirw ) ;
		sys[ind].calcCPChiCorrWrite( outfdirw ) ;
		sys[ind].calcPSxyCorrE     ( outfdirw ) ;
		sys[ind].calcPStotCorrE    ( outfdirw ) ;
		sys[ind].calcPStot         ( outfdirw ) ;
		sys[ind].calcPSxzStrCorrE   ( outfdirw ) ; // string corr
		sys[ind].calcPSxzSstrCorrE  ( outfdirw ) ; // string corr
		sys[ind].calcEE		   ( outfdirw ) ;
		sys[ind].write   	   ( outfnamew) ;
		sys[ind].saveNDelSparse    ( outfdirw ) ;

		sys[ind].freeInter() ;
		L++ ;
	}
	free(sys) ;

	sprintf( dum , "tar -czf %s %s* %s", outfnamec , outfnamew , outfdirw ) ;
	system( dum ) ;
	sprintf( dum , "scp %s dallae:%s", outfnamec , outdir ) ;
	system( dum ) ;
	sprintf( dum , "touch %s ", outfnametouch ) ;
	system( dum ) ;

	return 0 ;
}
