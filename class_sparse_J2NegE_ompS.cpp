#include <iostream>
#include <fstream>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>
#include <vector>

#ifndef CLASS_SPARSE_J2NEG_OMP_CPP
#define CLASS_SPARSE_J2NEG_OMP_CPP

class sparseDat {
	public  :
		size_t i,j ;
		double dat ;
		sparseDat( void ) {}
		sparseDat( size_t a, size_t b, double c ) {
			i=a ; j=b ; dat=c ;
		}
		void setDat( size_t a, size_t b, double c ) {
			i=a ; j=b ; dat=c ;
		}
		void cpyDat( sparseDat A ) {
			i=A.i ; j=A.j ; dat=A.dat ;
		}
} ;
class sparseMat {
	public  :
		size_t dim ,
			     sizeDat ;
		sparseDat *sDat ;
		sparseMat( void ) {
		}
		sparseMat( size_t d , size_t n ) {
			dim   = d ;
			sizeDat = n ;
			sDat = (sparseDat *) malloc( (sizeof(sparseDat)*sizeDat) ) ;
		}
		void setM( size_t d , size_t n ) {
			dim   = d ;
			sizeDat = n ;
			sDat = (sparseDat *) malloc( (sizeof(sparseDat)*sizeDat) ) ;
		}
		void sMatAllocCpy( sparseMat A ) {
			dim   = A.dim ;
			sizeDat = A.sizeDat ;
			sDat = (sparseDat *) malloc( (sizeof(sparseDat)*sizeDat) ) ;
			size_t d=0 ;
			for( d=0 ; d<sizeDat ; d++ ) {
				//sDat[d].i   = A.sDat[d].i ;
				//sDat[d].j   = A.sDat[d].j ;
				//sDat[d].dat = A.sDat[d].dat ;
				sDat[d].cpyDat( A.sDat[d] ) ;
			}
		}
		void sMatFree( void ) {
			dim=0 ;
			sizeDat=0 ;
			free( sDat ) ;
		}
		void prodSop( sparseMat A , sparseMat B ) {
			dim   = A.dim  * B.dim ;
			sizeDat = A.sizeDat * B.sizeDat ;
			sDat = (sparseDat *) malloc( (sizeof(sparseDat)*sizeDat) ) ;
			size_t ai=0 ,
				     bi=0 ;
			for( bi=0 ; bi<B.sizeDat ; bi++ ) {
				for( ai=0 ; ai<A.sizeDat ; ai++ ) {
					size_t d = ai + bi*( A.sizeDat ) ;
					sDat[ d ].i   = (A.sDat[ai].i) + A.dim * (B.sDat[bi].i) ;
					sDat[ d ].j   = (A.sDat[ai].j) + A.dim * (B.sDat[bi].j) ;
					sDat[ d ].dat = (A.sDat[ai].dat) * (B.sDat[bi].dat) ;
				}
			}
		}
		void prodSop( sparseMat A , sparseMat B, double scale ) {
			dim   = A.dim  * B.dim ;
			sizeDat = A.sizeDat * B.sizeDat ;
			sDat = (sparseDat *) malloc( (sizeof(sparseDat)*sizeDat) ) ;
			size_t ai=0 ,
				     bi=0 ;
			for( bi=0 ; bi<B.sizeDat ; bi++ ) {
				for( ai=0 ; ai<A.sizeDat ; ai++ ) {
					size_t d = ai + bi*( A.sizeDat ) ;
					sDat[ d ].i   = (A.sDat[ai].i) + A.dim * (B.sDat[bi].i) ;
					sDat[ d ].j   = (A.sDat[ai].j) + A.dim * (B.sDat[bi].j) ;
					sDat[ d ].dat = (A.sDat[ai].dat) * (B.sDat[bi].dat) * scale ;
				}
			}
		}
		void prodSopLId( sparseMat B, size_t idDim ) {
			//dum = "" ;
			dim   = idDim  * B.dim ;
			sizeDat = idDim * B.sizeDat ;
			sDat = (sparseDat *) malloc( (sizeof(sparseDat)*sizeDat) ) ;
			size_t ai=0 ,
				     bi=0 ;
			for( bi=0 ; bi<B.sizeDat ; bi++ ) {
				for( ai=0 ; ai<idDim ; ai++ ) {
					size_t d = ai + bi*( idDim ) ;
					sDat[ d ].i   = ai + idDim * (B.sDat[bi].i) ;
					sDat[ d ].j   = ai + idDim * (B.sDat[bi].j) ;
					sDat[ d ].dat = (B.sDat[bi].dat) ;
				}
			}
		}
		void prodSopRId( sparseMat A , size_t idDim ) {
			//dum = "" ;
			dim   = A.dim  * idDim ;
			sizeDat = A.sizeDat * idDim ;
			sDat = (sparseDat *) malloc( (sizeof(sparseDat)*sizeDat) ) ;
			size_t ai=0 ,
				     bi=0 ;
			for( bi=0 ; bi<idDim ; bi++ ) {
				for( ai=0 ; ai<A.sizeDat ; ai++ ) {
					size_t d = ai + bi*( A.sizeDat ) ;
					sDat[ d ].i   = (A.sDat[ai].i) + A.dim * bi ;
					sDat[ d ].j   = (A.sDat[ai].j) + A.dim * bi ;
					sDat[ d ].dat = (A.sDat[ai].dat) ;
				}
			}
		}
		void prodSop( sparseMat A , sparseMat B, sparseMat C, sparseMat D ) {
			sparseMat dumCD ,
				  dumBCD ,
				  dumABCD ;
			dumCD.prodSop( C , D ) ;
			dumBCD.prodSop( B , dumCD ) ;
			dumABCD.prodSop( A , dumBCD ) ;

			dim = dumABCD.dim ;
			sizeDat = dumABCD.sizeDat ; 
			sDat = dumABCD.sDat ;

			dumCD.sMatFree() ;
			dumBCD.sMatFree() ;
		}
		void prodSop2( sparseMat A , sparseMat B, sparseMat C, sparseMat D ) {
			dim   = A.dim  * B.dim * C.dim * D.dim ;
			sizeDat = A.sizeDat * B.sizeDat * C.sizeDat * D.sizeDat ;
			sDat = (sparseDat *) malloc( (sizeof(sparseDat)*sizeDat) ) ;
			size_t ai=0 ,
				     bi=0 ,
				     ci=0 ,
				     di=0 ;
			for( di=0 ; di<D.sizeDat ; di++ ) {
				for( ci=0 ; ci<C.sizeDat ; ci++ ) {
					for( bi=0 ; bi<B.sizeDat ; bi++ ) {
						for( ai=0 ; ai<A.sizeDat ; ai++ ) {
							size_t d = ai + bi*( A.sizeDat ) + ci*( A.sizeDat*B.sizeDat ) + di*( A.sizeDat*B.sizeDat*C.sizeDat ) ;
							sDat[ d ].i   =  (A.sDat[ai].i) + A.dim*(B.sDat[bi].i) + A.dim*B.dim*(C.sDat[ci].i) + A.dim*B.dim*C.dim*(D.sDat[di].i) ;
							sDat[ d ].j   =  (A.sDat[ai].j) + A.dim*(B.sDat[bi].j) + A.dim*B.dim*(C.sDat[ci].j) + A.dim*B.dim*C.dim*(D.sDat[di].j) ;
							sDat[ d ].dat = (A.sDat[ai].dat) * (B.sDat[bi].dat)* (C.sDat[ci].dat)* (D.sDat[di].dat) ;
						}
					}
				}
			}
		}
		void prodSop2( sparseMat B, sparseMat D, size_t idDimA , size_t idDimC ) {
			dim   = idDimA  * B.dim * idDimC * D.dim ;
			sizeDat = idDimA * B.sizeDat * idDimC * D.sizeDat ;
			sDat = (sparseDat *) malloc( (sizeof(sparseDat)*sizeDat) ) ;
			size_t ai=0 ,
				     bi=0 ,
				     ci=0 ,
				     di=0 ;
			for( di=0 ; di<D.sizeDat ; di++ ) {
				for( ci=0 ; ci<idDimC ; ci++ ) {
					for( bi=0 ; bi<B.sizeDat ; bi++ ) {
						for( ai=0 ; ai<idDimA ; ai++ ) {
							size_t d = ai + bi*( idDimA ) + ci*( idDimA*B.sizeDat ) + di*( idDimA*B.sizeDat*idDimC ) ;
							sDat[ d ].i   =  ai + idDimA*(B.sDat[bi].i) + idDimA*B.dim*ci + idDimA*B.dim*idDimC*(D.sDat[di].i) ;
							sDat[ d ].j   =  ai + idDimA*(B.sDat[bi].j) + idDimA*B.dim*ci + idDimA*B.dim*idDimC*(D.sDat[di].j) ;
							sDat[ d ].dat = (B.sDat[bi].dat)* (D.sDat[di].dat) ;
						}
					}
				}
			}
		}
		void prodSop( sparseMat A , sparseMat B, sparseMat C, sparseMat D, double scale ) {
			sparseMat dumCD ,
				  dumBCD ,
				  dumABCD ;
			dumCD.prodSop( C , D ) ;
			dumBCD.prodSop( B , dumCD ) ;
			dumABCD.prodSop( A , dumBCD, scale ) ;

			dim = dumABCD.dim ;
			sizeDat = dumABCD.sizeDat ; 
			sDat = dumABCD.sDat ;

			dumCD.sMatFree() ;
			dumBCD.sMatFree() ;
		}
		void prodSop2( sparseMat A , sparseMat B, sparseMat C, sparseMat D, double scale ) {
			dim   = A.dim  * B.dim * C.dim * D.dim ;
			sizeDat = A.sizeDat * B.sizeDat * C.sizeDat * D.sizeDat ;
			sDat = (sparseDat *) malloc( (sizeof(sparseDat)*sizeDat) ) ;
			size_t ai=0 ,
				     bi=0 ,
				     ci=0 ,
				     di=0 ;
			for( di=0 ; di<D.sizeDat ; di++ ) {
				for( ci=0 ; ci<C.sizeDat ; ci++ ) {
					for( bi=0 ; bi<B.sizeDat ; bi++ ) {
						for( ai=0 ; ai<A.sizeDat ; ai++ ) {
							size_t d = ai + bi*( A.sizeDat ) + ci*( A.sizeDat*B.sizeDat ) + di*( A.sizeDat*B.sizeDat*C.sizeDat ) ;
							sDat[ d ].i   =  (A.sDat[ai].i) + A.dim*(B.sDat[bi].i) + A.dim*B.dim*(C.sDat[ci].i) + A.dim*B.dim*C.dim*(D.sDat[di].i) ;
							sDat[ d ].j   =  (A.sDat[ai].j) + A.dim*(B.sDat[bi].j) + A.dim*B.dim*(C.sDat[ci].j) + A.dim*B.dim*C.dim*(D.sDat[di].j) ;
							sDat[ d ].dat = (A.sDat[ai].dat) * (B.sDat[bi].dat)* (C.sDat[ci].dat)* (D.sDat[di].dat) * scale ;
						}
					}
				}
			}
		}
		void prodSop2( sparseMat B, sparseMat D, size_t idDimA , size_t idDimC, double scale ) {
			dim   = idDimA  * B.dim * idDimC * D.dim ;
			sizeDat = idDimA * B.sizeDat * idDimC * D.sizeDat ;
			sDat = (sparseDat *) malloc( (sizeof(sparseDat)*sizeDat) ) ;
			size_t ai=0 ,
				     bi=0 ,
				     ci=0 ,
				     di=0 ;
			for( di=0 ; di<D.sizeDat ; di++ ) {
				for( ci=0 ; ci<idDimC ; ci++ ) {
					for( bi=0 ; bi<B.sizeDat ; bi++ ) {
						for( ai=0 ; ai<idDimA ; ai++ ) {
							size_t d = ai + bi*( idDimA ) + ci*( idDimA*B.sizeDat ) + di*( idDimA*B.sizeDat*idDimC ) ;
							sDat[ d ].i   =  ai + idDimA*(B.sDat[bi].i) + idDimA*B.dim*ci + idDimA*B.dim*idDimC*(D.sDat[di].i) ;
							sDat[ d ].j   =  ai + idDimA*(B.sDat[bi].j) + idDimA*B.dim*ci + idDimA*B.dim*idDimC*(D.sDat[di].j) ;
							sDat[ d ].dat = (B.sDat[bi].dat)* (D.sDat[di].dat) * scale ;
						}
					}
				}
			}
		}
		void addSop( sparseMat B ) {
			size_t ai=0 ,
				     bi=0 ;
			for( bi=0 ; bi<B.sizeDat ; bi++ ) {
				bool same =0 ;
				size_t sameInd=-1 ;
				for( ai=0 ; ai<sizeDat ; ai++ ) {
					if( sDat[ai].i == B.sDat[bi].i ) {
						if( sDat[ai].j == B.sDat[bi].j ) {
							same += 1 ;
							sameInd = ai ;
						}
					}
				}
				if( same > 0 ) {
					sDat[sameInd].dat += B.sDat[bi].dat ;
				}
				else {
					insertDat( B.sDat[bi] ) ;
				}
			}
		}
		void addSop( sparseMat A , sparseMat B ) {
			dim   = A.dim ; // =B.dim : two same-size square matrices
			sizeDat = A.sizeDat ; 
			sDat = (sparseDat *) malloc( (sizeof(sparseDat)*sizeDat) ) ;
			size_t ai=0 ,
				     bi=0 ;
			for( ai=0 ; ai<A.sizeDat ; ai++ ) {
				//sDat[ai].i   = A.sDat[ai].i ;
				//sDat[ai].j   = A.sDat[ai].j ;
				//sDat[ai].dat = A.sDat[ai].dat ;
				sDat[ai].setDat( A.sDat[ai].i , A.sDat[ai].j , A.sDat[ai].dat ) ;
			}
			for( bi=0 ; bi<B.sizeDat ; bi++ ) {
				bool same=0 ;
				size_t sameInd=-1 ;
				for( ai=0 ; ai<A.sizeDat ; ai++ ) {
					if( sDat[ai].i == B.sDat[bi].i ) {
						if( sDat[ai].j == B.sDat[bi].j ) {
							same += 1 ;
							sameInd = ai ;
						}
					}
				}
				if( same > 0 ) {
					sDat[sameInd].dat += B.sDat[bi].dat ;
				}
				else {
					insertDat( B.sDat[bi] ) ;
				}
			}
		}
		void addSop2( sparseMat B ) {
			size_t ai=0 ,
				     bi=0 ;
			sparseMat D(dim,0) ;
			std::vector<bool> aInd(sizeDat,0 ) ;

			for( bi=0 ; bi<B.sizeDat ; bi++ ) {
				bool same=0 ;
				size_t sameInd=-1 ;
				for( ai=0 ; ai<sizeDat ; ai++ ) {
					if( sDat[ai].i == B.sDat[bi].i ) {
						if( sDat[ai].j == B.sDat[bi].j ) {
							same += 1 ;
							sameInd = ai ;
						}
					}
				}
				if( same > 0 ) {
					double dum = sDat[sameInd].dat + B.sDat[bi].dat ;
					if( fabs(dum) > ZEROCUT ) {
						D.insertDat( sDat[sameInd].i , sDat[sameInd].j , dum  ) ;
					}
					aInd[sameInd] = 1 ;
				}
				else {
					D.insertDat( B.sDat[bi] ) ;
				}
			}
			for( ai=0 ; ai<sizeDat ; ai++ ) {
				if( aInd[ai] > 0 ) {
				}
				else {
					D.insertDat( sDat[ai] ) ;
				}
			}
			free( sDat ) ;
			sizeDat = D.sizeDat ;
			sDat = D.sDat ;
			aInd.clear() ;
		}
		void addSop2( sparseMat A , sparseMat B ) {
			dim   = A.dim ; // =B.dim : two same-size square matrices
			sizeDat = 0 ;
			sDat = NULL ;
			std::vector<bool> aInd(A.sizeDat,0) ;
			size_t ai=0 ,
				     bi=0 ;
			for( bi=0 ; bi<B.sizeDat ; bi++ ) {
				bool same =0 ;
				size_t sameInd=-1 ;
				for( ai=0 ; ai<A.sizeDat ; ai++ ) {
					if( A.sDat[ai].i == B.sDat[bi].i ) {
						if( A.sDat[ai].j == B.sDat[bi].j ) {
							same += 1 ;
							sameInd = ai ;
						}
					}
				}
				if( same > 0 ) {
					double dum = A.sDat[sameInd].dat + B.sDat[bi].dat ;
					if( fabs(dum) > ZEROCUT ) {
						insertDat( A.sDat[sameInd].i , A.sDat[sameInd].j , dum ) ;
					}
					aInd[sameInd] = 1 ; 
				}
				else {
					insertDat( B.sDat[bi] ) ;
				}
			}
			for( ai=0 ; ai<A.sizeDat ; ai++ ) {
				if( aInd[ai] > 0 ) {
				}
				else {
					insertDat( A.sDat[ai] ) ;
				}
			}
			aInd.clear() ;
		} 
		void addSop3( sparseMat B ) {
			sDat = (sparseDat *) realloc( sDat, sizeof(sparseDat)*(sizeDat+B.sizeDat) ) ;
			int **aInd2 = new int*[dim] ;
			for( size_t a=0 ; a<dim ; a++ ) 
				aInd2[a] = new int[dim] ;
			for( size_t a=0 ; a<dim ; a++ ) 
				for( size_t b=0 ; b<dim ; b++ ) 
					aInd2[a][b] = -1 ;
			for( size_t a=0 ; a<sizeDat ; a++ )
			{
				aInd2[ sDat[a].i ][ sDat[a].j ] = a ;
			}

			for( size_t bi=0 ; bi<B.sizeDat ; bi++ )
			{
				size_t row = B.sDat[bi].i ,
				       col = B.sDat[bi].j ;
				if( aInd2[ row ][ col ] < 0 )
				{
					sDat[sizeDat].setDat( row , col , B.sDat[bi].dat ) ;
					sizeDat += 1 ;
				}
				else
				{
					int dumI = aInd2[row][col] ;
					sDat[dumI].dat += B.sDat[bi].dat ;
				}
			}
			sDat = (sparseDat *) realloc( sDat, sizeof(sparseDat)*sizeDat ) ;
			for( size_t a=0 ; a<dim ; a++ )
				delete []aInd2[a] ;
			delete []aInd2 ;
		}
		void addSop3( sparseMat A , sparseMat B ) {
			sizeDat = A.sizeDat ;
			dim     = A.dim ;
			sDat = (sparseDat *) realloc( A.sDat, sizeof(sparseDat)*(sizeDat+B.sizeDat) ) ;
			int **aInd2 = new int*[dim] ;
			for( size_t a=0 ; a<dim ; a++ ) 
				aInd2[a] = new int[dim] ;
			for( size_t a=0 ; a<dim ; a++ ) 
				for( size_t b=0 ; b<dim ; b++ ) 
					aInd2[a][b] = -1 ;
			for( size_t a=0 ; a<sizeDat ; a++ )
			{
				aInd2[ A.sDat[a].i ][ A.sDat[a].j ] = a ;
			}

			for( size_t bi=0 ; bi<B.sizeDat ; bi++ )
			{
				size_t row = B.sDat[bi].i ,
				       col = B.sDat[bi].j ;
				if( aInd2[ row ][ col ] < 0 )
				{
					sDat[sizeDat].setDat( row , col , B.sDat[bi].dat ) ;
					sizeDat += 1 ;
				}
				else
				{
					int dumI = aInd2[row][col] ;
					sDat[dumI].dat += B.sDat[bi].dat ;
				}
			}
			sDat = (sparseDat *) realloc( sDat, sizeof(sparseDat)*sizeDat ) ;
			for( size_t a=0 ; a<dim ; a++ )
				delete []aInd2[a] ;
			delete []aInd2 ;
		}
		void scale( double sc ) {
			size_t d=0 ;
			for(d=0 ; d<sizeDat ; d++ ) {
				sDat[d].dat *= sc ;
			}
		}
		void insertDat( sparseDat n ) {
			sizeDat += 1 ;
			sDat = (sparseDat *) realloc( sDat, (sizeof(sparseDat)*sizeDat) ) ;
			sDat[sizeDat-1].setDat( n.i , n.j , n.dat ) ;
		}
		void insertDat( size_t i , size_t j , double d ) {
			sizeDat += 1 ;
			sDat = (sparseDat *) realloc( sDat, (sizeof(sparseDat)*sizeDat) ) ;
			sDat[sizeDat-1].setDat( i , j , d ) ;
		}
		void addDat( sparseDat n ) {
			bool same = 0 ;
			size_t sameInd=0 ;
			for( size_t ai=0 ; ai<sizeDat ; ai++ ) {
				if( sDat[ai].i == n.i ) {
					if( sDat[ai].j == n.j ) {
						same += 1 ;
						sameInd = ai ;
					}
				}
			}
			if( same > 0 ) {
				sDat[sameInd].dat += n.dat ;
			}
			else {
				insertDat( n ) ;
			}
		}
		void setIdSop( size_t a ) {
			dim = a ;
			sizeDat = a ;
			sDat = (sparseDat *) malloc( (sizeof(sparseDat)*sizeDat) ) ;
			for( size_t ai=0 ; ai<a ; ai++ ) {
				sDat[ai].setDat( ai, ai, 1 ) ;
			}
		}
		void dotGsl( gsl_vector *V , gsl_vector *Res ) {
			for( size_t ai=0 ; ai<sizeDat ; ai++ ) {
				gsl_vector_set( Res , sDat[ai].i , gsl_vector_get(Res,sDat[ai].i) + sDat[ai].dat * gsl_vector_get(V,sDat[ai].j) ) ;
			}
		}
		void TestdotGsl( gsl_matrix *H ) {
			for( size_t ai=0 ; ai<sizeDat ; ai++ ) {
				gsl_matrix_set( H , sDat[ai].i ,sDat[ai].j , gsl_matrix_get(H,sDat[ai].i,sDat[ai].j) + sDat[ai].dat ) ;
			}
		}
		void view( void ) {
			size_t d=0 ;
			for(d=0 ; d<sizeDat ; d++ ) {
				std::cout << "(" << sDat[d].i << "," << sDat[d].j << ")=" << sDat[d].dat << std::endl ;
			}
		}
		void view( std::ofstream &fs ) {
			size_t d=0 ;
			for(d=0 ; d<sizeDat ; d++ ) {
				fs << "(" << sDat[d].i << "," << sDat[d].j << ")=" << sDat[d].dat << std::endl ;
			}
		}
		void colOrd() {
			int forward  = (int)sizeDat-1 ;
			for( int max=forward ; forward>0 ; forward-- )
			{
				int ind=0 ;
				for( ; ind<max ; ind++ )
				{
					size_t dum=sDat[ind].j ;
					if( dum > sDat[ind+1].j )
					{
						size_t dumi = sDat[ind].i ;
						double dumd = sDat[ind].dat ;
						sDat[ind].i   = sDat[ind+1].i ;
						sDat[ind].j   = sDat[ind+1].j ;
						sDat[ind].dat = sDat[ind+1].dat ;

						sDat[ind+1].i   = dumi ;
						sDat[ind+1].j   = dum  ;
						sDat[ind+1].dat = dumd ;
					}
				}
			}
			forward  = (int)sizeDat-1 ;
			for( int max=forward ; forward>0 ; forward-- )
			{
				int ind=0 ;
				for( ; ind<max ; ind++ )
				{
					size_t dum=sDat[ind].j ;
					if( dum != sDat[ind+1].j ) {}
					else
					{
						dum = sDat[ind].i ;
						if( dum > sDat[ind+1].i )
						{
							size_t dumj = sDat[ind].j ;
							double dumd = sDat[ind].dat ;
							sDat[ind].i   = sDat[ind+1].i ;
							sDat[ind].j   = sDat[ind+1].j ;
							sDat[ind].dat = sDat[ind+1].dat ;

							sDat[ind+1].i   = dum  ;
							sDat[ind+1].j   = dumj ;
							sDat[ind+1].dat = dumd ;
						}
					}
				}
			}
		}
		void addSparse3( sparseMat &A, sparseMat &B ) 
		{ 
			size_t a=0,b=0,c=0; 

			setM( A.dim ,  A.sizeDat + B.sizeDat ) ;
			dim = A.dim ;
			sizeDat = 0 ;

			while( (a<A.sizeDat) && (b<B.sizeDat) ) 
			{ 
				if( A.sDat[a].j == B.sDat[b].j ) 
				{ 
					if( A.sDat[a].i == B.sDat[b].i ) 
					{ 
						sDat[c].i = A.sDat[a].i ;
						sDat[c].j = A.sDat[a].j ;
						sDat[c].dat = A.sDat[a].dat + B.sDat[b].dat ;
						sizeDat += 1 ;
						a++ ;
						b++ ;
						c++ ;
					} 
					else if( A.sDat[a].i < B.sDat[b].i ) 
					{ 
						sDat[c].i = A.sDat[a].i ;
						sDat[c].j = A.sDat[a].j ;
						sDat[c].dat = A.sDat[a].dat ;
						sizeDat += 1 ;
						a++ ;
						c++ ;
					} 
					else 
					{ 
						sDat[c].i = B.sDat[b].i ;
						sDat[c].j = B.sDat[b].j ;
						sDat[c].dat = B.sDat[b].dat ;
						sizeDat += 1 ;
						b++ ;
						c++ ;
					} 
				} 
				else if( A.sDat[a].j < B.sDat[b].j )
				{ 
					sDat[c].i = A.sDat[a].i ;
					sDat[c].j = A.sDat[a].j ;
					sDat[c].dat = A.sDat[a].dat ;
					sizeDat += 1 ;
					a++ ;
					c++ ;
				} 
				else 
				{ 
					sDat[c].i = B.sDat[b].i ;
					sDat[c].j = B.sDat[b].j ;
					sDat[c].dat = B.sDat[b].dat ;
					sizeDat += 1 ;
					b++ ;
					c++ ;
				} 
			} 
			while( a < A.sizeDat ) 
			{ 
				sDat[c].i = A.sDat[a].i ;
				sDat[c].j = A.sDat[a].j ;
				sDat[c].dat = A.sDat[a].dat ;
				sizeDat += 1 ;
				a++ ;
				c++ ;
			} 
			while( b < B.sizeDat ) 
			{ 
				sDat[c].i = B.sDat[b].i ;
				sDat[c].j = B.sDat[b].j ;
				sDat[c].dat = B.sDat[b].dat ;
				sizeDat += 1 ;
				b++ ;
				c++ ;
			} 
		} 
		void addSparse3( sparseMat &B ) 
		{ 
			sparseMat C ;
			size_t a=0,b=0,c=0; 

			C.setM( dim ,  sizeDat + B.sizeDat ) ;
			C.dim = dim ;
			C.sizeDat = 0 ;

			while( (a<sizeDat) && (b<B.sizeDat) ) 
			{ 
				if( sDat[a].j == B.sDat[b].j ) 
				{ 
					if( sDat[a].i == B.sDat[b].i ) 
					{ 
						C.sDat[c].i = sDat[a].i ;
						C.sDat[c].j = sDat[a].j ;
						C.sDat[c].dat = sDat[a].dat + B.sDat[b].dat ;
						C.sizeDat += 1 ;
						a++ ;
						b++ ;
						c++ ;
					} 
					else if( sDat[a].i < B.sDat[b].i ) 
					{ 
						C.sDat[c].i = sDat[a].i ;
						C.sDat[c].j = sDat[a].j ;
						C.sDat[c].dat = sDat[a].dat ;
						C.sizeDat += 1 ;
						a++ ;
						c++ ;
					} 
					else 
					{ 
						C.sDat[c].i = B.sDat[b].i ;
						C.sDat[c].j = B.sDat[b].j ;
						C.sDat[c].dat = B.sDat[b].dat ;
						C.sizeDat += 1 ;
						b++ ;
						c++ ;
					} 
				} 
				else if( sDat[a].j < B.sDat[b].j )
				{ 
					C.sDat[c].i = sDat[a].i ;
					C.sDat[c].j = sDat[a].j ;
					C.sDat[c].dat = sDat[a].dat ;
					C.sizeDat += 1 ;
					a++ ;
					c++ ;
				} 
				else 
				{ 
					C.sDat[c].i = B.sDat[b].i ;
					C.sDat[c].j = B.sDat[b].j ;
					C.sDat[c].dat = B.sDat[b].dat ;
					C.sizeDat += 1 ;
					b++ ;
					c++ ;
				} 
			} 
			while( a < sizeDat ) 
			{ 
				C.sDat[c].i = sDat[a].i ;
				C.sDat[c].j = sDat[a].j ;
				C.sDat[c].dat = sDat[a].dat ;
				C.sizeDat += 1 ;
				a++ ;
				c++ ;
			} 
			while( b < B.sizeDat ) 
			{ 
				C.sDat[c].i = B.sDat[b].i ;
				C.sDat[c].j = B.sDat[b].j ;
				C.sDat[c].dat = B.sDat[b].dat ;
				C.sizeDat += 1 ;
				b++ ;
				c++ ;
			} 
			free( sDat ) ;
			dim     = C.dim ;
			sizeDat = C.sizeDat ;
			sDat    = C.sDat ;
		} 
		void testnan(int &tnan)
		{
			for(size_t d=0 ; d<sizeDat ; d++ ) {
				if( isnan( sDat[d].dat ) )
					tnan++ ;
				//std::cout << "(" << sDat[d].i << "," << sDat[d].j << ")=" << sDat[d].dat << std::endl ;
			}
		}
		void testinf(int &tinf)
		{
			for(size_t d=0 ; d<sizeDat ; d++ ) {
				if( isinf( sDat[d].dat ) )
					tinf++ ;
			}
		}
} ;

class sopSpinOne {
	public  :
		sparseMat Sz ,
			  Sp ,
			  Sm ;
		sopSpinOne( void ) {
			Sz.setM(DIM,2) ;
			Sz.sDat[0].setDat( 0,0, 1 ) ;
			Sz.sDat[1].setDat( 2,2,-1 ) ;
			Sp.setM(DIM,2) ;
			Sp.sDat[0].setDat( 0,1,sqrt(2) ) ;
			Sp.sDat[1].setDat( 1,2,sqrt(2) ) ;
			Sm.setM(DIM,2) ;
			Sm.sDat[0].setDat( 1,0,sqrt(2) ) ;
			Sm.sDat[1].setDat( 2,1,sqrt(2) ) ;
			std::cout << "constructed\n" ;
		}
		void sopFree( void ) {
			Sz.sMatFree() ;
			Sp.sMatFree() ;
			Sm.sMatFree() ;
		}
} ;
class sopSpinHalfCoup {
	public  :
		sparseMat Sz ,
			  Sp ,
			  Sm ;
		sopSpinHalfCoup( void ) {
			Sz.setM(DIMH,2) ;
			Sz.sDat[0].setDat( 0,0, 0.5 ) ;
			Sz.sDat[1].setDat( 1,1,-0.5 ) ;
			Sp.setM(DIMH,1) ;
			Sp.sDat[0].setDat( 0,1, 1 ) ;
			Sm.setM(DIMH,1) ;
			Sm.sDat[0].setDat( 1,0, 1 ) ;
			std::cout << "constructed\n" ;
		}
		void sopFree( void ) {
			Sz.sMatFree() ;
			Sp.sMatFree() ;
			Sm.sMatFree() ;
		}
} ;
class sopComp {
	public  :
		sparseMat Sz ,
			  Sp ,
			  Sm ;
		void allocCpy( sopSpinOne a ) {
			Sz.sMatAllocCpy( a.Sz ) ;
			Sp.sMatAllocCpy( a.Sp ) ;
			Sm.sMatAllocCpy( a.Sm ) ;
		}
		void allocCpy( sopComp a ) {
			Sz.sMatAllocCpy( a.Sz ) ;
			Sp.sMatAllocCpy( a.Sp ) ;
			Sm.sMatAllocCpy( a.Sm ) ;
		}
		void allocCpy( sopSpinHalfCoup a ) {
			Sz.sMatAllocCpy( a.Sz ) ;
			Sp.sMatAllocCpy( a.Sp ) ;
			Sm.sMatAllocCpy( a.Sm ) ;
		}
		void sopCFree( void ) {
			Sz.sMatFree() ;
			Sp.sMatFree() ;
			Sm.sMatFree() ;
		}
} ;
class sparseMatOpform {
	public :
		sparseMat *S ,
			  **I ,
			  *I2 ;
		unsigned int dimA ,
			     dimB ,
			     dimAB ,
			     dimABC ;
		double Jz ;
		double J2 ;
		double J1n ;
		void alloc()
		{
			S = (sparseMat  *)malloc( sizeof(sparseMat )*2 ) ;
			I = (sparseMat **)malloc( sizeof(sparseMat*)*3 ) ;
			I2= (sparseMat  *)malloc( sizeof(sparseMat )*3 ) ;
			for( int i=0 ; i<3 ; i++ )
			{
				I[i] = (sparseMat *)malloc( sizeof(sparseMat)*2 ) ;
			}
		}
		void opfFree()
		{
			free( S ) ;
			for( int i=0 ; i<3 ; i++ )
			{
				free( I[i] ) ;
			}
			free( I ) ;
			free( I2) ;
		}
		void setDim ( size_t nBlock , size_t nSite ) 
		{
			dimA  = nBlock ;
			dimB  = nSite  ;
			dimAB = nBlock * nSite ;
			dimABC = dimAB * nBlock ;
		}
		void setJz( double JJz )
		{
			Jz = JJz ;
		}
		void setJ2( double JJ2 )
		{
			J2 = JJ2 ;
		}
		void set1( sparseMat &AB )
		{
			S[0] = AB ;
		}
		void set2( sparseMat &CD )
		{
			S[1] = CD ;
		}
		void setI( int n , sparseMat &B , sparseMat &D )
		{
			I[n][0] = B ;
			I[n][1] = D ;
		}
		void setI2( sparseMat &BSZ , sparseMat &BSP, sparseMat &BSM )
		{
			I2[0] = BSZ ;
			I2[1] = BSP ;
			I2[2] = BSM ;
		}
		void setAll ( size_t nbst, size_t nsst , double jz, double j2, sparseMat &sysH , sopComp &sS, sopComp &sB )
		{
			alloc() ;
			setDim( nbst, nsst ) ;
			setJz( jz ) ;
			setJ2( j2 ) ;
			set1 ( sysH ) ;
			set2 ( sysH ) ;
			setI ( 0 , sS.Sz , sS.Sz ) ;
			setI ( 1 , sS.Sp , sS.Sm ) ;
			setI ( 2 , sS.Sm , sS.Sp ) ;
			setI2( sB.Sz , sB.Sp , sB.Sm ) ;
		}
		void setAll ( size_t nbst, size_t nsst , double jz, double j2, sparseMat &sysH , sopComp &sS, sopComp &sB , double j1n )
		{
			alloc() ;
			setDim( nbst, nsst ) ;
			setJz( jz ) ;
			setJ2( j2 ) ;
			J1n = j1n ;
			set1 ( sysH ) ;
			set2 ( sysH ) ;
			setI ( 0 , sS.Sz , sS.Sz ) ;
			setI ( 1 , sS.Sp , sS.Sm ) ;
			setI ( 2 , sS.Sm , sS.Sp ) ;
			setI2( sB.Sz , sB.Sp , sB.Sm ) ;
		}
		void opfDotGslJ2( gsl_vector *V , gsl_vector *Res ) {
			//FOR S[0]'s (sysHamil)
			for( unsigned int bi=0 ; bi<dimAB ; bi++ ) {
				for( unsigned int ai=0 ; ai<S[0].sizeDat ; ai++ ) {
					size_t ii   = (S[0].sDat[ai].i) + dimAB * bi ;
					size_t jj   = (S[0].sDat[ai].j) + dimAB * bi ;
					gsl_vector_set( Res,ii , gsl_vector_get(Res,ii) + S[0].sDat[ai].dat*gsl_vector_get(V,jj) ) ;
				}
			}
			//FOR S[1]'s (envHamil=sysHamil)
			for( unsigned int bi=0 ; bi<S[1].sizeDat ; bi++ ) {
				for( unsigned int ai=0 ; ai<dimAB ; ai++ ) {
					size_t ii   = ai + dimAB * (S[1].sDat[bi].i) ;
					size_t jj   = ai + dimAB * (S[1].sDat[bi].j) ;
					gsl_vector_set( Res,ii , gsl_vector_get(Res,ii) + S[1].sDat[bi].dat*gsl_vector_get(V,jj) ) ;
				}
			}
			//FOR I[0]'s (ssiteSZ.ssiteSz)
			for( unsigned int di=0 ; di<I[0][0].sizeDat ; di++ ) {
				for( unsigned int ci=0 ; ci<dimA ; ci++ ) {
					for( unsigned int bi=0 ; bi<I[0][1].sizeDat ; bi++ ) {
						for( unsigned int ai=0 ; ai<dimA ; ai++ ) {
							size_t ii   =  ai + dimA*(I[0][1].sDat[bi].i) + dimAB*ci + dimABC*(I[0][0].sDat[di].i) ;
							size_t jj   =  ai + dimA*(I[0][1].sDat[bi].j) + dimAB*ci + dimABC*(I[0][0].sDat[di].j) ;
							gsl_vector_set( Res,ii ,
									gsl_vector_get(Res,ii) +
									Jz*(I[0][1].sDat[bi].dat * I[0][0].sDat[di].dat)*gsl_vector_get(V,jj) ) ;
						}
					}
				}
			}
			//FOR I[1,2]'s (ssiteSp.ssiteSm + ssiteSm.ssiteSp)
			for( short int a=1 ; a<3 ; a++ )
			{
				for( unsigned int di=0 ; di<I[a][0].sizeDat ; di++ ) {
					for( unsigned int ci=0 ; ci<dimA ; ci++ ) {
						for( unsigned int bi=0 ; bi<I[a][1].sizeDat ; bi++ ) {
							for( unsigned int ai=0 ; ai<dimA ; ai++ ) {
								size_t ii   =  ai + dimA*(I[a][1].sDat[bi].i) + dimAB*ci + dimABC*(I[a][0].sDat[di].i) ;
								size_t jj   =  ai + dimA*(I[a][1].sDat[bi].j) + dimAB*ci + dimABC*(I[a][0].sDat[di].j) ;
								gsl_vector_set( Res,ii ,
										gsl_vector_get(Res,ii) +
										0.5*(I[a][1].sDat[bi].dat * I[a][0].sDat[di].dat)*gsl_vector_get(V,jj) ) ;
							}
						}
					}
				}
			}

			//For J2 ( sysBlocksiteS{z,p,m}.lssiteS{z,m.p} + rssiteS{z,p,m}.envBlocksiteS{z,m,p} )
			for( unsigned int di=0 ; di<I[0][0].sizeDat ; di++ ) {
				for( unsigned int ci=0 ; ci<dimA ; ci++ ) {
					for( unsigned int bi=0 ; bi<dimB ; bi++ ) {
						for( unsigned int ai=0 ; ai<I2[0].sizeDat ; ai++ ) {
							size_t ii   = I2[0].sDat[ai].i + dimA*bi  + dimAB*ci + dimABC*(I[0][0].sDat[di].i) ;
							size_t jj   = I2[0].sDat[ai].j + dimA*bi  + dimAB*ci + dimABC*(I[0][0].sDat[di].j) ;
							gsl_vector_set( Res,ii ,
									gsl_vector_get(Res,ii) +
									J2*Jz*(I2[0].sDat[ai].dat * I[0][0].sDat[di].dat)*gsl_vector_get(V,jj) ) ;
						}
					}
				}
			}
			for( unsigned int di=0 ; di<I[2][0].sizeDat ; di++ ) {
				for( unsigned int ci=0 ; ci<dimA ; ci++ ) {
					for( unsigned int bi=0 ; bi<dimB ; bi++ ) {
						for( unsigned int ai=0 ; ai<I2[1].sizeDat ; ai++ ) {
							size_t ii   = I2[1].sDat[ai].i + dimA*bi  + dimAB*ci + dimABC*(I[2][0].sDat[di].i) ;
							size_t jj   = I2[1].sDat[ai].j + dimA*bi  + dimAB*ci + dimABC*(I[2][0].sDat[di].j) ;
							gsl_vector_set( Res,ii ,
									gsl_vector_get(Res,ii) +
									J2*0.5*(I2[1].sDat[ai].dat * I[2][0].sDat[di].dat)*gsl_vector_get(V,jj) ) ;
						}
					}
				}
			}
			for( unsigned int di=0 ; di<I[1][0].sizeDat ; di++ ) {
				for( unsigned int ci=0 ; ci<dimA ; ci++ ) {
					for( unsigned int bi=0 ; bi<dimB ; bi++ ) {
						for( unsigned int ai=0 ; ai<I2[2].sizeDat ; ai++ ) {
							size_t ii   = I2[2].sDat[ai].i + dimA*bi  + dimAB*ci + dimABC*(I[1][0].sDat[di].i) ;
							size_t jj   = I2[2].sDat[ai].j + dimA*bi  + dimAB*ci + dimABC*(I[1][0].sDat[di].j) ;
							gsl_vector_set( Res,ii ,
									gsl_vector_get(Res,ii) +
									J2*0.5*(I2[2].sDat[ai].dat * I[1][0].sDat[di].dat)*gsl_vector_get(V,jj) ) ;
						}
					}
				}
			}
			for( unsigned int di=0 ; di<dimB ; di++ ) {
				for( unsigned int ci=0 ; ci<I2[0].sizeDat ; ci++ ) {
					for( unsigned int bi=0 ; bi<I[0][0].sizeDat ; bi++ ) {
						for( unsigned int ai=0 ; ai<dimA ; ai++ ) {
							size_t ii   = ai + dimA*(I[0][0].sDat[bi].i)  + dimAB*I2[0].sDat[ci].i  + dimABC*di ;
							size_t jj   = ai + dimA*(I[0][0].sDat[bi].j)  + dimAB*I2[0].sDat[ci].j  + dimABC*di ;
							gsl_vector_set( Res,ii ,
									gsl_vector_get(Res,ii) +
									J2*Jz*(I2[0].sDat[ci].dat * I[0][0].sDat[bi].dat)*gsl_vector_get(V,jj) ) ;
						}
					}
				}
			}
			for( unsigned int di=0 ; di<dimB ; di++ ) {
				for( unsigned int ci=0 ; ci<I2[2].sizeDat ; ci++ ) {
					for( unsigned int bi=0 ; bi<I[1][0].sizeDat ; bi++ ) {
						for( unsigned int ai=0 ; ai<dimA ; ai++ ) {
							size_t ii   = ai + dimA*(I[1][0].sDat[bi].i)  + dimAB*I2[2].sDat[ci].i  + dimABC*di ;
							size_t jj   = ai + dimA*(I[1][0].sDat[bi].j)  + dimAB*I2[2].sDat[ci].j  + dimABC*di ;
							gsl_vector_set( Res,ii ,
									gsl_vector_get(Res,ii) +
									J2*0.5*(I2[2].sDat[ci].dat * I[1][0].sDat[bi].dat)*gsl_vector_get(V,jj) ) ;
						}
					}
				}
			}
			for( unsigned int di=0 ; di<dimB ; di++ ) {
				for( unsigned int ci=0 ; ci<I2[1].sizeDat ; ci++ ) {
					for( unsigned int bi=0 ; bi<I[2][0].sizeDat ; bi++ ) {
						for( unsigned int ai=0 ; ai<dimA ; ai++ ) {
							size_t ii   = ai + dimA*(I[2][0].sDat[bi].i)  + dimAB*I2[1].sDat[ci].i  + dimABC*di ;
							size_t jj   = ai + dimA*(I[2][0].sDat[bi].j)  + dimAB*I2[1].sDat[ci].j  + dimABC*di ;
							gsl_vector_set( Res,ii ,
									gsl_vector_get(Res,ii) +
									J2*0.5*(I2[1].sDat[ci].dat * I[2][0].sDat[bi].dat)*gsl_vector_get(V,jj) ) ;
						}
					}
				}
			}
		}
		void opfDotGslJ2Neg( gsl_vector *V , gsl_vector *Res ) {
			//FOR S[0]'s (sysHamil)
#pragma omp parallel for shared(Res) schedule(static)
			for( unsigned int bi=0 ; bi<dimAB ; bi++ ) {
				for( unsigned int ai=0 ; ai<S[0].sizeDat ; ai++ ) {
					size_t ii   = (S[0].sDat[ai].i) + dimAB * bi ;
					size_t jj   = (S[0].sDat[ai].j) + dimAB * bi ;
					gsl_vector_set( Res,ii , gsl_vector_get(Res,ii) + S[0].sDat[ai].dat*gsl_vector_get(V,jj) ) ;
				}
			}
			//FOR S[1]'s (envHamil=sysHamil)
#pragma omp parallel for shared(Res) schedule(static)
			for( unsigned int ai=0 ; ai<dimAB ; ai++ ) {
				for( unsigned int bi=0 ; bi<S[1].sizeDat ; bi++ ) {
					size_t ii   = ai + dimAB * (S[1].sDat[bi].i) ;
					size_t jj   = ai + dimAB * (S[1].sDat[bi].j) ;
					gsl_vector_set( Res,ii , gsl_vector_get(Res,ii) + S[1].sDat[bi].dat*gsl_vector_get(V,jj) ) ;
				}
			}
			//FOR I[0]'s (ssiteSZ.ssiteSz)
#pragma omp parallel for shared(Res) schedule(static)
			for( unsigned int ci=0 ; ci<dimA ; ci++ ) {
				for( unsigned int di=0 ; di<I[0][0].sizeDat ; di++ ) {
					for( unsigned int bi=0 ; bi<I[0][1].sizeDat ; bi++ ) {
						for( unsigned int ai=0 ; ai<dimA ; ai++ ) {
							size_t ii   =  ai + dimA*(I[0][1].sDat[bi].i) + dimAB*ci + dimABC*(I[0][0].sDat[di].i) ;
							size_t jj   =  ai + dimA*(I[0][1].sDat[bi].j) + dimAB*ci + dimABC*(I[0][0].sDat[di].j) ;
							gsl_vector_set( Res,ii ,
									gsl_vector_get(Res,ii) +
									J1n*Jz*(I[0][1].sDat[bi].dat * I[0][0].sDat[di].dat)*gsl_vector_get(V,jj) ) ;
						}
					}
				}
			}
			//FOR I[1,2]'s (ssiteSp.ssiteSm + ssiteSm.ssiteSp)
			for( short int a=1 ; a<3 ; a++ )
			{
#pragma omp parallel for shared(Res) schedule(static)
				for( unsigned int ci=0 ; ci<dimA ; ci++ ) {
					for( unsigned int di=0 ; di<I[a][0].sizeDat ; di++ ) {
						for( unsigned int bi=0 ; bi<I[a][1].sizeDat ; bi++ ) {
							for( unsigned int ai=0 ; ai<dimA ; ai++ ) {
								size_t ii   =  ai + dimA*(I[a][1].sDat[bi].i) + dimAB*ci + dimABC*(I[a][0].sDat[di].i) ;
								size_t jj   =  ai + dimA*(I[a][1].sDat[bi].j) + dimAB*ci + dimABC*(I[a][0].sDat[di].j) ;
								gsl_vector_set( Res,ii ,
										gsl_vector_get(Res,ii) +
										J1n*0.5*(I[a][1].sDat[bi].dat * I[a][0].sDat[di].dat)*gsl_vector_get(V,jj) ) ;
							}
						}
					}
				}
			}

			//For J2 ( sysBlocksiteS{z,p,m}.rssiteS{z,m.p} + lssiteS{z,p,m}.envBlocksiteS{z,m,p} )
#pragma omp parallel for shared(Res) schedule(static)
			for( unsigned int ci=0 ; ci<dimA ; ci++ ) {
				for( unsigned int di=0 ; di<I[0][0].sizeDat ; di++ ) {
					for( unsigned int bi=0 ; bi<dimB ; bi++ ) {
						for( unsigned int ai=0 ; ai<I2[0].sizeDat ; ai++ ) {
							size_t ii   = I2[0].sDat[ai].i + dimA*bi  + dimAB*ci + dimABC*(I[0][0].sDat[di].i) ;
							size_t jj   = I2[0].sDat[ai].j + dimA*bi  + dimAB*ci + dimABC*(I[0][0].sDat[di].j) ;
							gsl_vector_set( Res,ii ,
									gsl_vector_get(Res,ii) +
									J2*Jz*(I2[0].sDat[ai].dat * I[0][0].sDat[di].dat)*gsl_vector_get(V,jj) ) ;
						}
					}
				}
			}
#pragma omp parallel for shared(Res) schedule(static)
			for( unsigned int ci=0 ; ci<dimA ; ci++ ) {
				for( unsigned int di=0 ; di<I[2][0].sizeDat ; di++ ) {
					for( unsigned int bi=0 ; bi<dimB ; bi++ ) {
						for( unsigned int ai=0 ; ai<I2[1].sizeDat ; ai++ ) {
							size_t ii   = I2[1].sDat[ai].i + dimA*bi  + dimAB*ci + dimABC*(I[2][0].sDat[di].i) ;
							size_t jj   = I2[1].sDat[ai].j + dimA*bi  + dimAB*ci + dimABC*(I[2][0].sDat[di].j) ;
							gsl_vector_set( Res,ii ,
									gsl_vector_get(Res,ii) +
									J2*0.5*(I2[1].sDat[ai].dat * I[2][0].sDat[di].dat)*gsl_vector_get(V,jj) ) ;
						}
					}
				}
			}
#pragma omp parallel for shared(Res) schedule(static)
			for( unsigned int ci=0 ; ci<dimA ; ci++ ) {
				for( unsigned int di=0 ; di<I[1][0].sizeDat ; di++ ) {
					for( unsigned int bi=0 ; bi<dimB ; bi++ ) {
						for( unsigned int ai=0 ; ai<I2[2].sizeDat ; ai++ ) {
							size_t ii   = I2[2].sDat[ai].i + dimA*bi  + dimAB*ci + dimABC*(I[1][0].sDat[di].i) ;
							size_t jj   = I2[2].sDat[ai].j + dimA*bi  + dimAB*ci + dimABC*(I[1][0].sDat[di].j) ;
							gsl_vector_set( Res,ii ,
									gsl_vector_get(Res,ii) +
									J2*0.5*(I2[2].sDat[ai].dat * I[1][0].sDat[di].dat)*gsl_vector_get(V,jj) ) ;
						}
					}
				}
			}
#pragma omp parallel for shared(Res) schedule(static)
			for( unsigned int ai=0 ; ai<dimA ; ai++ ) {
				for( unsigned int di=0 ; di<dimB ; di++ ) {
					for( unsigned int ci=0 ; ci<I2[0].sizeDat ; ci++ ) {
						for( unsigned int bi=0 ; bi<I[0][0].sizeDat ; bi++ ) {
							size_t ii   = ai + dimA*(I[0][0].sDat[bi].i)  + dimAB*I2[0].sDat[ci].i  + dimABC*di ;
							size_t jj   = ai + dimA*(I[0][0].sDat[bi].j)  + dimAB*I2[0].sDat[ci].j  + dimABC*di ;
							gsl_vector_set( Res,ii ,
									gsl_vector_get(Res,ii) +
									J2*Jz*(I2[0].sDat[ci].dat * I[0][0].sDat[bi].dat)*gsl_vector_get(V,jj) ) ;
						}
					}
				}
			}
#pragma omp parallel for shared(Res) schedule(static)
			for( unsigned int ai=0 ; ai<dimA ; ai++ ) {
				for( unsigned int di=0 ; di<dimB ; di++ ) {
					for( unsigned int ci=0 ; ci<I2[2].sizeDat ; ci++ ) {
						for( unsigned int bi=0 ; bi<I[1][0].sizeDat ; bi++ ) {
							size_t ii   = ai + dimA*(I[1][0].sDat[bi].i)  + dimAB*I2[2].sDat[ci].i  + dimABC*di ;
							size_t jj   = ai + dimA*(I[1][0].sDat[bi].j)  + dimAB*I2[2].sDat[ci].j  + dimABC*di ;
							gsl_vector_set( Res,ii ,
									gsl_vector_get(Res,ii) +
									J2*0.5*(I2[2].sDat[ci].dat * I[1][0].sDat[bi].dat)*gsl_vector_get(V,jj) ) ;
						}
					}
				}
			}
#pragma omp parallel for shared(Res) schedule(static)
			for( unsigned int ai=0 ; ai<dimA ; ai++ ) {
				for( unsigned int di=0 ; di<dimB ; di++ ) {
					for( unsigned int ci=0 ; ci<I2[1].sizeDat ; ci++ ) {
						for( unsigned int bi=0 ; bi<I[2][0].sizeDat ; bi++ ) {
							size_t ii   = ai + dimA*(I[2][0].sDat[bi].i)  + dimAB*I2[1].sDat[ci].i  + dimABC*di ;
							size_t jj   = ai + dimA*(I[2][0].sDat[bi].j)  + dimAB*I2[1].sDat[ci].j  + dimABC*di ;
							gsl_vector_set( Res,ii ,
									gsl_vector_get(Res,ii) +
									J2*0.5*(I2[1].sDat[ci].dat * I[2][0].sDat[bi].dat)*gsl_vector_get(V,jj) ) ;
						}
					}
				}
			}
		}
} ;
class sparseMatOpformOb {
	public :
		sopComp *totS ;
		sopComp cenS ;
		sparseMat *POpSz ,
			  *POpSS ;
		unsigned int dimA ,
			     dimB ,
			     dimAB ,
			     dimABC ,
			     L ;
		double *Sz ,
		       *localSS ,
		       *PLocalSS ;
		void alloc()
		{
			L=2 ;
			totS  =   (sopComp *)malloc( sizeof(sopComp)  *L ) ;
			Sz    = (double*)malloc( sizeof(double)*2*L ) ;
			localSS= (double*)malloc( sizeof(double)*(2*L-1) ) ;
		}
		void allocPOp( int n )
		{
			L=n ;
			POpSz = (sparseMat *)malloc( sizeof(sparseMat)  * L    ) ;
			POpSS = (sparseMat *)malloc( sizeof(sparseMat)  *(L-1) ) ;
			Sz    = (double*)malloc( sizeof(double)*2*L ) ;
			PLocalSS= (double*)malloc( sizeof(double)*(2*L-1) ) ;
		}
		void allocPOpSz( int n )
		{
			L=n ;
			POpSz = (sparseMat *)malloc( sizeof(sparseMat)  * L    ) ;
			Sz    = (double*)malloc( sizeof(double)*2*L ) ;
		}
		void reallocPOp( int n )
		{
			L=n ;
			POpSz = (sparseMat *)realloc( POpSz , sizeof(sparseMat)  * L    ) ;
			POpSS = (sparseMat *)realloc( POpSS , sizeof(sparseMat)  *(L-1) ) ;
			Sz    = (double*)realloc( Sz, sizeof(double)*2*L ) ;
			PLocalSS=(double*)realloc(PLocalSS, sizeof(double)*(2*L-1) ) ;
		}
		void allocTotS( int n )
		{
			L=n ;
			totS  =   (sopComp *)malloc( sizeof(sopComp)  *L ) ;
			Sz    = (double*)malloc( sizeof(double)*2*L ) ;
			localSS= (double*)malloc( sizeof(double)*(2*L-1) ) ;
		}
		void reallocTotS( int n )
		{
			L=n ;
			totS  = (sopComp *)realloc( totS , sizeof(sopComp)*L ) ;
			Sz    = (double*)realloc( Sz, sizeof(double)*2*L ) ;
			localSS=(double*)realloc(localSS, sizeof(double)*(2*L-1) ) ;
		}
		void freePOpSS()
		{
			for( int a=0 ; a<(int)L-1 ; a++ ) {
				POpSS[a].sMatFree() ;
			}
			free( POpSS ) ;
			free( PLocalSS ) ;
		}
		void freePOpSz()
		{
			for( int a=0 ; a<(int)L-3 ; a++ ) {
				POpSz[a].sMatFree() ;
			}
			free( POpSz ) ;
			free( Sz ) ;
		}
		void freePOp()
		{
			for( int a=0 ; a<(int)L-3 ; a++ ) {
				POpSz[a].sMatFree() ;
			}

			free( POpSz ) ;
			free( POpSS ) ;
			free( Sz ) ;
			free( PLocalSS ) ;
		}
		void freeTotS()
		{
			for( int a=0 ; a<(int)L-2 ; a++ ) {
				totS[a].Sz.sMatFree() ;
			}
			free( totS ) ;
			free( Sz   ) ;
			free( localSS) ;
		}
		void freeTotSAll()
		{
			for( int a=0 ; a<(int)L-2 ; a++ ) {
				totS[a].Sz.sMatFree() ;
				totS[a].Sp.sMatFree() ;
				totS[a].Sm.sMatFree() ;
			}
			free( totS ) ;
			free( Sz   ) ;
			free( localSS) ;
		}
		void opfFree()
		{
			free( totS ) ;
		}
		void setTotS ( sopComp &S1 , int i )
		{
			totS[i] = S1 ;
		}
		void setDim ( size_t nBlock , size_t nSite ) 
		{
			dimA  = nBlock ;
			dimB  = nSite  ;
			dimAB = nBlock * nSite ;
			dimABC = dimAB * nBlock ;
		}
		void PDdotGsl( sparseMat &M, gsl_vector *v1 , gsl_vector *v2 , short int ptype )
		{
			switch( ptype ) 
			{
				case 0 :
#pragma omp parallel for shared(v2) schedule(static)
					for( unsigned int ci=0 ; ci<dimA ; ci++ ) {
						for( unsigned int di=0 ; di<dimB ; di++ ) {
							for( unsigned int bi=0 ; bi<dimB ; bi++ ) {
								for( unsigned int ai=0 ; ai<M.sizeDat ; ai++ ) {
									size_t ii   =  M.sDat[ai].i + dimA*(bi) + dimAB*ci + dimABC*di ;
									size_t jj   =  M.sDat[ai].j + dimA*(bi) + dimAB*ci + dimABC*di ;
									gsl_vector_set( v2,ii , gsl_vector_get(v2,ii) + M.sDat[ai].dat * gsl_vector_get(v1,jj) ) ;
								}
							}
						}
					}
					break ;
				case 1 :
#pragma omp parallel for shared(v2) schedule(static)
					for( unsigned int ci=0 ; ci<dimA ; ci++ ) {
						for( unsigned int di=0 ; di<dimB ; di++ ) {
							for( unsigned int bi=0 ; bi<M.sizeDat ; bi++ ) {
								for( unsigned int ai=0 ; ai<dimA ; ai++ ) {
									size_t ii   =  ai + dimA*(M.sDat[bi].i) + dimAB*ci + dimABC*di ;
									size_t jj   =  ai + dimA*(M.sDat[bi].j) + dimAB*ci + dimABC*di ;
									gsl_vector_set( v2,ii , gsl_vector_get(v2,ii) + M.sDat[bi].dat * gsl_vector_get(v1,jj) ) ;
								}
							}
						}
					}
					break ;
				case 2 :
#pragma omp parallel for shared(v2) schedule(static)
					for( unsigned int ai=0 ; ai<dimA ; ai++ ) {
						for( unsigned int di=0 ; di<dimB ; di++ ) {
							for( unsigned int ci=0 ; ci<M.sizeDat ; ci++ ) {
								for( unsigned int bi=0 ; bi<dimB ; bi++ ) {
									size_t ii   =  ai + dimA*(bi) + dimAB*M.sDat[ci].i + dimABC*di ;
									size_t jj   =  ai + dimA*(bi) + dimAB*M.sDat[ci].j + dimABC*di ;
									gsl_vector_set( v2,ii , gsl_vector_get(v2,ii) + M.sDat[ci].dat * gsl_vector_get(v1,jj) ) ;
								}
							}
						}
					}
					break ;
				case 3 :
#pragma omp parallel for shared(v2) schedule(static)
					for( unsigned int ai=0 ; ai<dimA ; ai++ ) {
						for( unsigned int bi=0 ; bi<dimB ; bi++ ) {
							for( unsigned int ci=0 ; ci<dimA ; ci++ ) {
								for( unsigned int di=0 ; di<M.sizeDat ; di++ ) {
									size_t ii   =  ai + dimA*(bi) + dimAB*ci + dimABC*M.sDat[di].i  ;
									size_t jj   =  ai + dimA*(bi) + dimAB*ci + dimABC*M.sDat[di].j  ;
									gsl_vector_set( v2,ii , gsl_vector_get(v2,ii) + M.sDat[di].dat * gsl_vector_get(v1,jj) ) ;
								}
							}
						}
					}
					break ;
				case 4 :
#pragma omp parallel for shared(v2) schedule(static)
					for( unsigned int ci=0 ; ci<dimA ; ci++ ) {
						for( unsigned int di=0 ; di<dimB ; di++ ) {
							for( unsigned int abi=0 ; abi<M.sizeDat ; abi++ ) {
								size_t ii   =  M.sDat[abi].i + dimAB*ci + dimABC*di ;
								size_t jj   =  M.sDat[abi].j + dimAB*ci + dimABC*di ;
								gsl_vector_set( v2,ii , gsl_vector_get(v2,ii) + M.sDat[abi].dat * gsl_vector_get(v1,jj) ) ;
							}
						}
					}
					break ;
				case 5 :
#pragma omp parallel for shared(v2) schedule(static)
					for( unsigned int ai=0 ; ai<dimA ; ai++ ) {
						for( unsigned int bi=0 ; bi<dimB ; bi++ ) {
							for( unsigned int cdi=0 ; cdi<M.sizeDat ; cdi++ ) {
									size_t ii   =  ai + dimA*(bi) + dimAB*M.sDat[cdi].i  ;
									size_t jj   =  ai + dimA*(bi) + dimAB*M.sDat[cdi].j  ;
									gsl_vector_set( v2,ii , gsl_vector_get(v2,ii) + M.sDat[cdi].dat * gsl_vector_get(v1,jj) ) ;
								}
							}
					}
					break ;
			}
		}
		double PDExpectSz( gsl_vector *GS , int a , int b )
		{
			gsl_vector *Res = gsl_vector_calloc( GS->size ) ;
			double dum ;
			PDdotGsl( totS[a].Sz , GS, Res, b ) ;
			gsl_blas_ddot( GS, Res,   &dum ) ;
			gsl_vector_free(Res) ;
			return dum ;
		}
		double PDExpectPOpSz( gsl_vector *GS , int a , int b )
		{
			gsl_vector *Res = gsl_vector_calloc( GS->size ) ;
			double dum ;
			PDdotGsl( POpSz[a] , GS, Res, b ) ;
			gsl_blas_ddot( GS, Res,   &dum ) ;
			gsl_vector_free(Res) ;
			return dum ;
		}
		void POpSzGsl( gsl_vector *GS ) {
			
			int dum = 2*L-1 ;
			for( unsigned int a=0 ; a<L-1 ; a++ ) 
			{
				Sz[a]     = PDExpectPOpSz( GS , a , 0 ) ;
				Sz[dum-a] = PDExpectPOpSz( GS , a , 2 ) ;
			}
			Sz[L-1] = PDExpectPOpSz( GS , L-1 , 1 ) ;
			Sz[L]   = PDExpectPOpSz( GS , L-1 , 3 ) ;

		}
		void SzGsl( gsl_vector *GS ) {
			
			int dum = 2*L-1 ;
			for( unsigned int a=0 ; a<L-1 ; a++ ) 
			{
				Sz[a]     = PDExpectSz( GS , a , 0 ) ;
				Sz[dum-a] = PDExpectSz( GS , a , 2 ) ;
			}
			Sz[L-1] = PDExpectSz( GS , L-1 , 1 ) ;
			Sz[L]   = PDExpectSz( GS , L-1 , 3 ) ;

		}
		double POpSzCorrGslSym( gsl_vector *GS , int nstart )
		{
			gsl_vector *Res  = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res2 = gsl_vector_calloc( GS->size ) ;
			double dum ;
			if( nstart < (int)L-1 )
			{
				PDdotGsl( POpSz[nstart] , GS , Res , 0 ) ;
				PDdotGsl( POpSz[nstart] , Res, Res2, 2 ) ;
			}
			else
			{
				PDdotGsl( POpSz[nstart] , GS , Res , 1 ) ;
				PDdotGsl( POpSz[nstart] , Res, Res2, 3 ) ;
			}
			gsl_blas_ddot( GS, Res2, &dum ) ;

			gsl_vector_free(Res ) ;
			gsl_vector_free(Res2) ;
			return dum ;
		}
		double SzCorrGslSym( gsl_vector *GS , int nstart )
		{
			gsl_vector *Res  = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res2 = gsl_vector_calloc( GS->size ) ;
			double dum ;
			if( nstart < (int)L-1 )
			{
				PDdotGsl( totS[nstart].Sz , GS , Res , 0 ) ;
				PDdotGsl( totS[nstart].Sz , Res, Res2, 2 ) ;
			}
			else
			{
				PDdotGsl( totS[nstart].Sz , GS , Res , 1 ) ;
				PDdotGsl( totS[nstart].Sz , Res, Res2, 3 ) ;
			}
			gsl_blas_ddot( GS, Res2, &dum ) ;

			gsl_vector_free(Res ) ;
			gsl_vector_free(Res2) ;
			return dum ;
		}
		void PLocalSSGsl( gsl_vector *GS , gsl_vector *R, int pos, int SSpos, int ind1 )
		{
			gsl_vector_set_zero( R ) ;
			PDdotGsl( POpSS[pos] , GS , R, ind1 ) ;

			gsl_blas_ddot( GS, R, &PLocalSS[SSpos] ) ;
		}
		void PLocalSSGsl2( gsl_vector *GS , gsl_vector *R, gsl_vector *R2 , int SSpos , double JJJz )
		{
			gsl_vector_set_zero( R2) ;
			gsl_vector_set_zero( R ) ;
			PDdotGsl( cenS.Sz , GS , R , 3 ) ; gsl_vector_scale( R, JJJz ) ;
			PDdotGsl( cenS.Sz , R  , R2, 1 ) ;

			gsl_vector_set_zero( R ) ;
			PDdotGsl( cenS.Sm , GS , R , 3 ) ; gsl_vector_scale( R, 0.5 ) ;
			PDdotGsl( cenS.Sp , R  , R2, 1 ) ;

			gsl_vector_set_zero( R ) ;
			PDdotGsl( cenS.Sp , GS , R , 3 ) ; gsl_vector_scale( R, 0.5 ) ;
			PDdotGsl( cenS.Sm , R  , R2, 1 ) ;

			gsl_blas_ddot( GS, R2, &PLocalSS[SSpos] ) ;
		}
		void localSSGsl( gsl_vector *GS , gsl_vector *R, gsl_vector *R2 , int pos, int SSpos, int ind1 , int ind2 )
		{
			gsl_vector_set_zero( R2) ;
			gsl_vector_set_zero( R ) ;
			PDdotGsl( totS[pos+1].Sz , GS , R , ind2 ) ;
			PDdotGsl( totS[pos  ].Sz , R  , R2, ind1 ) ;

			gsl_vector_set_zero( R ) ;
			PDdotGsl( totS[pos+1].Sm , GS , R , ind2 ) ; gsl_vector_scale( R, 0.5 ) ;
			PDdotGsl( totS[pos  ].Sp , R  , R2, ind1 ) ;

			gsl_vector_set_zero( R ) ;
			PDdotGsl( totS[pos+1].Sp , GS , R , ind2 ) ; gsl_vector_scale( R, 0.5 ) ;
			PDdotGsl( totS[pos  ].Sm , R  , R2, ind1 ) ;

			gsl_blas_ddot( GS, R2, &localSS[SSpos] ) ;
		}
		void localSSGsl2( gsl_vector *GS , gsl_vector *R, gsl_vector *R2 , int pos, int pos2, int SSpos, int ind1 , int ind2 )
		{
			gsl_vector_set_zero( R2) ;
			gsl_vector_set_zero( R ) ;
			PDdotGsl( totS[pos2 ].Sz , GS , R , ind2 ) ;
			PDdotGsl( totS[pos  ].Sz , R  , R2, ind1 ) ;

			gsl_vector_set_zero( R ) ;
			PDdotGsl( totS[pos2 ].Sm , GS , R , ind2 ) ; gsl_vector_scale( R, 0.5 ) ;
			PDdotGsl( totS[pos  ].Sp , R  , R2, ind1 ) ;

			gsl_vector_set_zero( R ) ;
			PDdotGsl( totS[pos2 ].Sp , GS , R , ind2 ) ; gsl_vector_scale( R, 0.5 ) ;
			PDdotGsl( totS[pos  ].Sm , R  , R2, ind1 ) ;

			gsl_blas_ddot( GS, R2, &localSS[SSpos] ) ;
		}
		void iterLocalSSGsl( gsl_vector *GS )
		{
			gsl_vector *Res  = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res2 = gsl_vector_calloc( GS->size ) ;
			int totLink = 2*L-2 ;
			for( int n=0 ; n<(int)L-2 ; n++ )
			{
				localSSGsl( GS , Res , Res2 , n , n , 0 , 0 ) ;
			}

			//Environment's
			for( int n=0 ; n<(int)L-2 ; n++ )
			{
				localSSGsl( GS , Res , Res2 , n , totLink-n , 2 , 2 ) ;
			}

			// n=L-2 case :
			localSSGsl ( GS , Res , Res2 , L-2 , L-2 , 0 , 1 ) ;
			// n=L-1 case :
			localSSGsl2( GS , Res , Res2 , L-1 , L-1 , L-1 , 1 , 3 ) ;
			// n=L case :
			localSSGsl2( GS , Res , Res2 , L-1 , L-2 , L   , 3 , 2 ) ;

			gsl_vector_free(Res ) ;
			gsl_vector_free(Res2) ;
		}
		void iterPLocalSSGsl( gsl_vector *GS , double JJz )
		{
			gsl_vector *Res  = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res2 = gsl_vector_calloc( GS->size ) ;
			int totLink = 2*L-2 ;
			//System Block's
			for( int n=0 ; n<(int)L-2 ; n++ )
			{
				PLocalSSGsl( GS , Res , n , n , 0 ) ;
			}

			//Environment Block's
			for( int n=0 ; n<(int)L-2 ; n++ )
			{
				PLocalSSGsl( GS , Res , n , totLink-n , 2 ) ;
			}
			PLocalSSGsl( GS , Res , L-2 , L-2 , 4 ) ;
			PLocalSSGsl( GS , Res , L-2 , L   , 5 ) ;
			PLocalSSGsl2( GS , Res , Res2 , L-1 , JJz ) ;

			gsl_vector_free(Res ) ;
			gsl_vector_free(Res2) ;
		}
		double SsqGsl( gsl_vector *GS ) {
			gsl_vector *Res = gsl_vector_calloc( GS->size ) ,
				   *vdum= gsl_vector_calloc( GS->size ) ;
			double totSsqVal ;

			for( unsigned int i=0 ; i<2 ; i++ ) 
			{
				for( unsigned int a=0 ; a<L-1 ; a++ ) 
				{
					PDdotGsl( totS[a].Sz , GS, vdum , i ) ;
					PDdotGsl( totS[a].Sz , vdum, Res, i ) ;

					gsl_vector_set_zero( vdum ) ;
					PDdotGsl( totS[a].Sp , GS, vdum , i ) ;
					gsl_vector_scale( vdum , 0.5 ) ;
					PDdotGsl( totS[a].Sm , vdum, Res, i ) ;

					gsl_vector_set_zero( vdum ) ;
					PDdotGsl( totS[a].Sm , GS, vdum , i ) ;
					gsl_vector_scale( vdum , 0.5 ) ;
					PDdotGsl( totS[a].Sp , vdum, Res, i ) ;
				}
			}
			int a=L-1 ;
			for( unsigned int i=2 ; i<4 ; i++ ) 
			{
				PDdotGsl( totS[a].Sz , GS, vdum , i ) ;
				PDdotGsl( totS[a].Sz , vdum, Res, i ) ;

				gsl_vector_set_zero( vdum ) ;
				PDdotGsl( totS[a].Sp , GS, vdum , i ) ;
				gsl_vector_scale( vdum , 0.5 ) ;
				PDdotGsl( totS[a].Sm , vdum, Res, i ) ;

				gsl_vector_set_zero( vdum ) ;
				PDdotGsl( totS[a].Sm , GS, vdum , i ) ;
				gsl_vector_scale( vdum , 0.5 ) ;
				PDdotGsl( totS[a].Sp , vdum, Res, i ) ;
			}

			gsl_blas_ddot( GS, Res,   &totSsqVal ) ;
			gsl_vector_free( Res ) ;
			return totSsqVal ;
		}
} ;

class sparseMatOpformObChi : public sparseMatOpformOb 
{
	public :
		sparseMat *POpChi ;
		double *ChiCorr ;
		void setDimL ( size_t length , size_t nBlock , size_t nSite ) 
		{
			dimA  = nBlock ;
			dimB  = nSite  ;
			dimAB = nBlock * nSite ;
			dimABC = dimAB * nBlock ;
			L = length ;
		}
		void allocPOpSz()
		{
			POpSz = (sparseMat *)malloc( sizeof(sparseMat)  * L    ) ;
			Sz    = (double*    )malloc( sizeof(double)     * 2*L  ) ;
		}
		void allocPOpChi()
		{
			POpChi     = (sparseMat *)malloc( sizeof(sparseMat)  *(L-1  ) ) ;
			ChiCorr    = (double*    )malloc( sizeof(double)     *(2*L-3) ) ;
		}
		void allocPOpSS()
		{
			POpSS      = (sparseMat *)malloc( sizeof(sparseMat)  *(L-1  ) ) ;
			PLocalSS   = (double*    )malloc( sizeof(double)     *(2*L-1) ) ;
		}
		void initPOpSz( sparseMat &sssz , sparseMat &sbsz , sopComp &sb2 )
		{
			allocPOpSz () ;
			POpSz[1] = sssz ; 
			POpSz[0] = sbsz ; 
			sb2.Sz.setM( dimA , 0 ) ;
			sb2.Sp.setM( dimA , 0 ) ;
			sb2.Sm.setM( dimA , 0 ) ;
		}
		void initPOpChi( sopComp &ss , sopComp &sb )
		{
			allocPOpChi() ;
			POpChi[0].prodSop( sb.Sp, ss.Sm , 0.5 ) ;
			sparseMat dum ;
			dum.prodSop      ( sb.Sm, ss.Sp ,-0.5 ) ;
			POpChi[0].addSop3( dum ) ;
			dum.sMatFree() ;
		}
		void updatePOpSz( sparseMat &sssz , sparseMat &sbsz , sparseMat &sb2sz )
		{
			allocPOpSz () ;
			POpSz[L-1] = sssz ; 
			POpSz[L-2] = sbsz ; 
			POpSz[L-3] = sb2sz ; 
		}
		void updatePOpChi( sopComp ss, sopComp sb ) 
		{
			allocPOpChi() ;
			POpChi[L-2].prodSop( sb.Sp, ss.Sm , 0.5 ) ;
			sparseMat dum ;
			dum.prodSop        ( sb.Sm, ss.Sp ,-0.5 ) ;
			POpChi[L-2].addSop3( dum ) ;
			dum.sMatFree() ;
		}
		void freePOpChi()
		{
			for( int a=0 ; a<(int)L-1 ; a++ ) 
			{
				POpChi[a].sMatFree() ;
			}
			free( POpChi ) ;
			free( ChiCorr    ) ;
		}
		double POpChiCorrGslSym( gsl_vector *GS , int oopInd , int l )
		{
			gsl_vector *Res  = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res2 = gsl_vector_calloc( GS->size ) ;
			double dum ;
			if( l < 1 )
			{
				PDdotGsl( POpChi[oopInd] , GS , Res , 4 ) ;
				PDdotGsl( POpChi[oopInd] , Res, Res2, 5 ) ;
			}
			else
			{
				PDdotGsl( POpChi[oopInd] , GS , Res , 0 ) ;
				PDdotGsl( POpChi[oopInd] , Res, Res2, 2 ) ;
			}
			gsl_blas_ddot( GS, Res2, &dum ) ;

			gsl_vector_free(Res ) ;
			gsl_vector_free(Res2) ;
			return dum ;
		}
		double POpChiCorrGslAsym( gsl_vector *GS , int oopInd , int l )
		{
			gsl_vector *Res  = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res2 = gsl_vector_calloc( GS->size ) ;
			double dum ;
			if( l < 1 )
			{
				PDdotGsl( POpChi[oopInd  ] , GS , Res , 4 ) ;
				PDdotGsl( POpChi[oopInd-1] , Res, Res2, 2 ) ;
			}
			else
			{
				PDdotGsl( POpChi[oopInd  ] , GS , Res , 0 ) ;
				PDdotGsl( POpChi[oopInd-1] , Res, Res2, 2 ) ;
			}
			gsl_blas_ddot( GS, Res2, &dum ) ;

			gsl_vector_free(Res ) ;
			gsl_vector_free(Res2) ;
			return dum ;
		}
} ;
class sparseMatOpformObChiS : public sparseMatOpformObChi
{
	public :
		sparseMat *POpSp ,
			  *POpSm ;
		sparseMat *POpStrSz ,
			  *POpStrSx ,
			  *POpSstrSz ,
			  *POpSstrSx ;
		void allocPOpStrSx()
		{
			POpStrSx = (sparseMat *) malloc( sizeof(sparseMat) * (L-1)       ) ;
		}
		void allocPOpStrSz()
		{
			POpStrSz = (sparseMat *) malloc( sizeof(sparseMat) * (L-1)       ) ;
		}
		void allocPOpSstrSx()
		{
			POpSstrSx = (sparseMat *) malloc( sizeof(sparseMat) * L ) ;
		}
		void allocPOpSstrSz()
		{
			POpSstrSz = (sparseMat *) malloc( sizeof(sparseMat) * L ) ;
		}
		void allocPOpSxy()
		{
			POpSp = (sparseMat *) malloc( sizeof(sparseMat) * L       ) ;
			POpSm = (sparseMat *) malloc( sizeof(sparseMat) * L       ) ;
		}
		void freePOpSxy()
		{
			for( int a=0 ; a<int(L)-3 ; a++ ) {
				POpSp[a].sMatFree() ;
				POpSm[a].sMatFree() ;
			}
			free( POpSp ) ;
			free( POpSm ) ;
		}
		void initPOpSxy( sopComp &ss, sopComp &sb ) 
		{
			allocPOpSxy() ;
			POpSp[1] = ss.Sp ;
			POpSm[1] = ss.Sm ;
			POpSp[0] = sb.Sp ;
			POpSm[0] = sb.Sm ;
		}
		void initPOpSstr( sopComp &ss, sopComp &sb ) 
		{
			allocPOpSstrSz() ;
			POpSstrSz[L-1] = ss.Sz ;
			POpSstrSz[L-2].sMatAllocCpy( sb.Sz ) ;

			allocPOpSstrSx() ;
			sparseMat sssx ,
				  sbsx ;
			sssx.sMatAllocCpy( ss.Sp ) ;
			sssx.addSop3( ss.Sm ) ;
			sssx.scale(0.5) ;
			sbsx.sMatAllocCpy( sb.Sp ) ;
			sbsx.addSop3( sb.Sm ) ;
			sbsx.scale(0.5) ;
			POpSstrSx[L-1] = sssx ;
			POpSstrSx[L-2] = sbsx ;
		}
		void freePOpSstrSx()
		{
			for( int a=0 ; a<int(L) ; a++ ) 
			{
				POpSstrSx[a].sMatFree() ;
			}
			free( POpSstrSx ) ;
		}
		void freePOpSstrSz()
		{
			for( int a=0 ; a<int(L)-1 ; a++ ) 
			{
				POpSstrSz[a].sMatFree() ;
			}
			free( POpSstrSz ) ;
		}
		void initPOpStr( sopComp &ss, sopComp &sb ) 
		{
			allocPOpStrSz() ;
			POpStrSz[0].prodSop( sb.Sz, ss.Sz ) ;

			allocPOpStrSx() ;
			sparseMat sssx ,
				  sbsx ;
			sssx.sMatAllocCpy( ss.Sp ) ;
			sssx.addSop3( ss.Sm ) ;
			sbsx.sMatAllocCpy( sb.Sp ) ;
			sbsx.addSop3( sb.Sm ) ;
			POpStrSx[0].prodSop( sbsx, sssx , 0.25 ) ;
			sssx.sMatFree() ;
			sbsx.sMatFree() ;
		}
		void updatePOpStr( sopComp &ss, sopComp &sb ) 
		{
			allocPOpStrSz() ;
			POpStrSz[L-2].prodSop( sb.Sz, ss.Sz ) ;

			allocPOpStrSx() ;
			sparseMat sssx ,
				  sbsx ;
			sssx.sMatAllocCpy( ss.Sp ) ;
			sssx.addSop3( ss.Sm ) ;
			sbsx.sMatAllocCpy( sb.Sp ) ;
			sbsx.addSop3( sb.Sm ) ;
			POpStrSx[L-2].prodSop( sbsx, sssx , 0.25 ) ;
			sssx.sMatFree() ;
			sbsx.sMatFree() ;
		}
		void freePOpStrSx()
		{
			for( int a=0 ; a<(int)L-1 ; a++ ) 
			{
				POpStrSx[a].sMatFree() ;
			}
			free( POpStrSx ) ;
		}
		void freePOpStrSz()
		{
			for( int a=0 ; a<(int)L-1 ; a++ ) 
			{
				POpStrSz[a].sMatFree() ;
			}
			free( POpStrSz ) ;
		}
		double POpSxyCorrGslSym( gsl_vector *GS , int nstart ) 
		{
			gsl_vector *Res  = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res2 = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res3 = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res4 = gsl_vector_calloc( GS->size ) ;
			double dum ;
			if( nstart < (int)L-1 )
			{    
				PDdotGsl( POpSp[nstart] , GS , Res , 0 ) ;
				PDdotGsl( POpSm[nstart] , Res, Res2, 2 ) ;

				PDdotGsl( POpSm[nstart] , GS  , Res3, 0 ) ;
				PDdotGsl( POpSp[nstart] , Res3, Res4, 2 ) ;
			}    
			else 
			{    
				PDdotGsl( POpSp[nstart] , GS , Res , 1 ) ;
				PDdotGsl( POpSm[nstart] , Res, Res2, 3 ) ;

				PDdotGsl( POpSm[nstart] , GS  , Res3, 1 ) ;
				PDdotGsl( POpSp[nstart] , Res3, Res4, 3 ) ;
			}    
			gsl_vector_add  ( Res2, Res4 ) ;
			gsl_vector_scale( Res2, 0.5  ) ;
			gsl_blas_ddot( GS, Res2, &dum ) ;

			gsl_vector_free(Res ) ;
			gsl_vector_free(Res2) ;
			gsl_vector_free(Res3) ;
			gsl_vector_free(Res4) ;
			return dum ;
		}
		double POpSxyCorrGslSymE( gsl_vector *GS , int nstart ) 
		{
			gsl_vector *Res  = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res2 = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res3 = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res4 = gsl_vector_calloc( GS->size ) ;
			double dum ;
			if( nstart < (int)L-1 )
			{    
				PDdotGsl( POpSp[nstart  ] , GS , Res , 0 ) ;
				PDdotGsl( POpSm[nstart-1] , Res, Res2, 2 ) ;

				PDdotGsl( POpSm[nstart  ] , GS  , Res3, 0 ) ;
				PDdotGsl( POpSp[nstart-1] , Res3, Res4, 2 ) ;
			}    
			else 
			{    
				PDdotGsl( POpSp[nstart  ] , GS , Res , 1 ) ;
				PDdotGsl( POpSm[nstart-1] , Res, Res2, 2 ) ;

				PDdotGsl( POpSm[nstart  ] , GS  , Res3, 1 ) ;
				PDdotGsl( POpSp[nstart-1] , Res3, Res4, 2 ) ;
			}    
			gsl_vector_add  ( Res2, Res4 ) ;
			gsl_vector_scale( Res2, 0.5  ) ;
			gsl_blas_ddot( GS, Res2, &dum ) ;

			gsl_vector_free(Res ) ;
			gsl_vector_free(Res2) ;
			gsl_vector_free(Res3) ;
			gsl_vector_free(Res4) ;
			return dum ;
		}
		double POpStotCorrGslSym( gsl_vector *GS , int nstart ) 
		{
			gsl_vector *Res  = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res2 = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res3 = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res4 = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res5 = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res6 = gsl_vector_calloc( GS->size ) ;
			double dum ;
			if( nstart < (int)L-1 )
			{    
				PDdotGsl( POpSp[nstart] , GS , Res , 0 ) ;
				PDdotGsl( POpSm[nstart] , Res, Res2, 2 ) ;

				PDdotGsl( POpSm[nstart] , GS  , Res3, 0 ) ;
				PDdotGsl( POpSp[nstart] , Res3, Res4, 2 ) ;

				PDdotGsl( POpSz[nstart] , GS  , Res5, 0 ) ;
				PDdotGsl( POpSz[nstart] , Res5, Res6, 2 ) ;
			}    
			else 
			{    
				PDdotGsl( POpSp[nstart] , GS , Res , 1 ) ;
				PDdotGsl( POpSm[nstart] , Res, Res2, 3 ) ;

				PDdotGsl( POpSm[nstart] , GS  , Res3, 1 ) ;
				PDdotGsl( POpSp[nstart] , Res3, Res4, 3 ) ;

				PDdotGsl( POpSz[nstart] , GS  , Res5, 1 ) ;
				PDdotGsl( POpSz[nstart] , Res5, Res6, 3 ) ;
			}    
			gsl_vector_add  ( Res2, Res4 ) ;
			gsl_vector_scale( Res2, 0.5  ) ;
			gsl_vector_add  ( Res2, Res6 ) ;
			gsl_blas_ddot( GS, Res2, &dum ) ;

			gsl_vector_free(Res ) ;
			gsl_vector_free(Res2) ;
			gsl_vector_free(Res3) ;
			gsl_vector_free(Res4) ;
			gsl_vector_free(Res5) ;
			gsl_vector_free(Res6) ;
			return dum ;
		}
		double POpStotCorrGslSymE( gsl_vector *GS , int nstart ) 
		{
			gsl_vector *Res  = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res2 = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res3 = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res4 = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res5 = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res6 = gsl_vector_calloc( GS->size ) ;
			double dum ;
			if( nstart < (int)L-1 )
			{    
				PDdotGsl( POpSp[nstart  ] , GS , Res , 0 ) ;
				PDdotGsl( POpSm[nstart-1] , Res, Res2, 2 ) ;

				PDdotGsl( POpSm[nstart  ] , GS  , Res3, 0 ) ;
				PDdotGsl( POpSp[nstart-1] , Res3, Res4, 2 ) ;

				PDdotGsl( POpSz[nstart  ] , GS  , Res5, 0 ) ;
				PDdotGsl( POpSz[nstart-1] , Res5, Res6, 2 ) ;
			}    
			else 
			{    
				PDdotGsl( POpSp[nstart]   , GS , Res , 1 ) ;
				PDdotGsl( POpSm[nstart-1] , Res, Res2, 2 ) ;

				PDdotGsl( POpSm[nstart]   , GS  , Res3, 1 ) ;
				PDdotGsl( POpSp[nstart-1] , Res3, Res4, 2 ) ;

				PDdotGsl( POpSz[nstart]   , GS  , Res5, 1 ) ;
				PDdotGsl( POpSz[nstart-1] , Res5, Res6, 2 ) ;
			}    
			gsl_vector_add  ( Res2, Res4 ) ;
			gsl_vector_scale( Res2, 0.5  ) ;
			gsl_vector_add  ( Res2, Res6 ) ;
			gsl_blas_ddot( GS, Res2, &dum ) ;

			gsl_vector_free(Res ) ;
			gsl_vector_free(Res2) ;
			gsl_vector_free(Res3) ;
			gsl_vector_free(Res4) ;
			gsl_vector_free(Res5) ;
			gsl_vector_free(Res6) ;
			return dum ;
		}
		double POpStot( gsl_vector *GS )
		{
			gsl_vector *Res  = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Resf = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Resx = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Resy = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Resz = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Resx2= gsl_vector_calloc( GS->size ) ;
			gsl_vector *Resy2= gsl_vector_calloc( GS->size ) ;
			gsl_vector *Resz2= gsl_vector_calloc( GS->size ) ;
			double dum ;
			for( size_t i=0 ; i<L ; i++ )
			{
				if( i < L-1 )
				{    
					gsl_vector_set_zero( Res ) ;
					PDdotGsl( POpSp[i] , GS  , Resx, 0 ) ;
					PDdotGsl( POpSp[i] , GS  , Resy, 0 ) ;
					PDdotGsl( POpSm[i] , GS  , Res , 0 ) ;
					PDdotGsl( POpSz[i] , GS  , Resz, 0 ) ;
					gsl_vector_add  ( Resx, Res  ) ;
					gsl_vector_sub  ( Resy, Res  ) ;

					gsl_vector_set_zero( Res ) ;
					PDdotGsl( POpSp[i] , GS  , Resx, 2 ) ;
					PDdotGsl( POpSp[i] , GS  , Resy, 2 ) ;
					PDdotGsl( POpSm[i] , GS  , Res , 2 ) ;
					PDdotGsl( POpSz[i] , GS  , Resz, 2 ) ;
					gsl_vector_add  ( Resx, Res  ) ;
					gsl_vector_sub  ( Resy, Res  ) ;
				}    
				else 
				{    
					gsl_vector_set_zero( Res ) ;
					PDdotGsl( POpSp[i] , GS  , Resx, 1 ) ;
					PDdotGsl( POpSp[i] , GS  , Resy, 1 ) ;
					PDdotGsl( POpSm[i] , GS  , Res , 1 ) ;
					PDdotGsl( POpSz[i] , GS  , Resz, 1 ) ;
					gsl_vector_add  ( Resx, Res  ) ;
					gsl_vector_sub  ( Resy, Res  ) ;

					gsl_vector_set_zero( Res ) ;
					PDdotGsl( POpSp[i] , GS  , Resx, 3 ) ;
					PDdotGsl( POpSp[i] , GS  , Resy, 3 ) ;
					PDdotGsl( POpSm[i] , GS  , Res , 3 ) ;
					PDdotGsl( POpSz[i] , GS  , Resz, 3 ) ;
					gsl_vector_add  ( Resx, Res  ) ;
					gsl_vector_sub  ( Resy, Res  ) ;
				}    
			}
			for( size_t i=0 ; i<L ; i++ )
			{
				if( i < L-1 )
				{    
					gsl_vector_set_zero( Res ) ;
					PDdotGsl( POpSp[i] , Resx, Resx2, 0 ) ;
					PDdotGsl( POpSm[i] , Resx, Resx2, 0 ) ;
					PDdotGsl( POpSp[i] , Resy, Resy2, 0 ) ;
					PDdotGsl( POpSm[i] , Resy, Res  , 0 ) ;
					PDdotGsl( POpSz[i] , Resz, Resf , 0 ) ;
					gsl_vector_sub  ( Resy2, Res  ) ;

					gsl_vector_set_zero( Res ) ;
					PDdotGsl( POpSp[i] , Resx, Resx2, 2 ) ;
					PDdotGsl( POpSm[i] , Resx, Resx2, 2 ) ;
					PDdotGsl( POpSp[i] , Resy, Resy2, 2 ) ;
					PDdotGsl( POpSm[i] , Resy, Res  , 2 ) ;
					PDdotGsl( POpSz[i] , Resz, Resf , 2 ) ;
					gsl_vector_sub  ( Resy2, Res  ) ;
				}    
				else 
				{    
					gsl_vector_set_zero( Res ) ;
					PDdotGsl( POpSp[i] , Resx, Resx2, 1 ) ;
					PDdotGsl( POpSm[i] , Resx, Resx2, 1 ) ;
					PDdotGsl( POpSp[i] , Resy, Resy2, 1 ) ;
					PDdotGsl( POpSm[i] , Resy, Res  , 1 ) ;
					PDdotGsl( POpSz[i] , Resz, Resf , 1 ) ;
					gsl_vector_sub  ( Resy2, Res  ) ;

					gsl_vector_set_zero( Res ) ;
					PDdotGsl( POpSp[i] , Resx, Resx2, 3 ) ;
					PDdotGsl( POpSm[i] , Resx, Resx2, 3 ) ;
					PDdotGsl( POpSp[i] , Resy, Resy2, 3 ) ;
					PDdotGsl( POpSm[i] , Resy, Res  , 3 ) ;
					PDdotGsl( POpSz[i] , Resz, Resf , 3 ) ;
					gsl_vector_sub  ( Resy2, Res  ) ;
				}    
			}
			gsl_vector_scale( Resx2, 0.25  ) ;
			gsl_vector_scale( Resy2, 0.25  ) ;
			gsl_vector_add  ( Resf, Resx2 ) ;
			gsl_vector_sub  ( Resf, Resy2 ) ;
			gsl_blas_ddot( GS, Resf, &dum ) ;

			gsl_vector_free(Res ) ;
			gsl_vector_free(Resf) ;
			gsl_vector_free(Resx) ;
			gsl_vector_free(Resy) ;
			gsl_vector_free(Resz) ;
			gsl_vector_free(Resx2) ;
			gsl_vector_free(Resy2) ;
			gsl_vector_free(Resz2) ;
			return dum ;
		}
		void POpStot2( gsl_vector *GS , double *stotcomp )
		{
			gsl_vector *Res  = gsl_vector_calloc( GS->size ) ,
				   *Resx = gsl_vector_calloc( GS->size ) ,
				   *Resy = gsl_vector_calloc( GS->size ) ,
				   *Resz = gsl_vector_calloc( GS->size ) ;
			for( size_t i=0 ; i<L ; i++ )
			{
				if( i < L-1 )
				{    
					gsl_vector_set_zero( Res ) ;
					PDdotGsl( POpSp[i] , GS  , Resx, 0 ) ;
					PDdotGsl( POpSp[i] , GS  , Resy, 0 ) ;
					PDdotGsl( POpSm[i] , GS  , Res , 0 ) ;
					PDdotGsl( POpSz[i] , GS  , Resz, 0 ) ;
					gsl_vector_add  ( Resx, Res  ) ;
					gsl_vector_sub  ( Resy, Res  ) ;

					gsl_vector_set_zero( Res ) ;
					PDdotGsl( POpSp[i] , GS  , Resx, 2 ) ;
					PDdotGsl( POpSp[i] , GS  , Resy, 2 ) ;
					PDdotGsl( POpSm[i] , GS  , Res , 2 ) ;
					PDdotGsl( POpSz[i] , GS  , Resz, 2 ) ;
					gsl_vector_add  ( Resx, Res  ) ;
					gsl_vector_sub  ( Resy, Res  ) ;
				}    
				else 
				{    
					gsl_vector_set_zero( Res ) ;
					PDdotGsl( POpSp[i] , GS  , Resx, 1 ) ;
					PDdotGsl( POpSp[i] , GS  , Resy, 1 ) ;
					PDdotGsl( POpSm[i] , GS  , Res , 1 ) ;
					PDdotGsl( POpSz[i] , GS  , Resz, 1 ) ;
					gsl_vector_add  ( Resx, Res  ) ;
					gsl_vector_sub  ( Resy, Res  ) ;

					gsl_vector_set_zero( Res ) ;
					PDdotGsl( POpSp[i] , GS  , Resx, 3 ) ;
					PDdotGsl( POpSp[i] , GS  , Resy, 3 ) ;
					PDdotGsl( POpSm[i] , GS  , Res , 3 ) ;
					PDdotGsl( POpSz[i] , GS  , Resz, 3 ) ;
					gsl_vector_add  ( Resx, Res  ) ;
					gsl_vector_sub  ( Resy, Res  ) ;
				}    
			}
			gsl_vector_free(Res ) ;

			gsl_vector_scale( Resx, 0.5  ) ;
			gsl_vector_scale( Resy, 0.5  ) ;
			gsl_blas_ddot( Resx , Resx , &stotcomp[0] ) ;
			gsl_blas_ddot( Resy , Resy , &stotcomp[1] ) ;
			gsl_blas_ddot( Resz , Resz , &stotcomp[2] ) ;

			gsl_vector_free(Resx) ;
			gsl_vector_free(Resy) ;
			gsl_vector_free(Resz) ;
		}
		void updatePOpSxy( sopComp &ss , sopComp &sb, sopComp &sb2 )
		{
			allocPOpSxy() ;
			POpSp[L-1] = ss.Sp ;
			POpSm[L-1] = ss.Sm ;
			POpSp[L-2] = sb.Sp ;
			POpSm[L-2] = sb.Sm ;
			POpSp[L-3] = sb2.Sp ;
			POpSm[L-3] = sb2.Sm ;
		}
		void POpSzStrCorrGslSym( gsl_vector *GS , int l , size_t L , double *a )
		{
			gsl_vector *Res  = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res2 = gsl_vector_calloc( GS->size ) ;
			int nstr = l/2 ;
			if( l < 1 )
			{
				PDdotGsl( POpSz[L-2] , GS , Res , 2 ) ;
				PDdotGsl( POpSz[L-1] , Res, Res2, 3 ) ;
				gsl_vector_free( Res ) ;
				Res = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpSz[L-1] , Res2, Res , 1 ) ;
				gsl_vector_free( Res2 ) ;
				Res2 = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpSz[L-2] , Res, Res2, 0 ) ;
				gsl_blas_ddot( GS, Res2, &a[0] ) ;
				a[1] = 0 ;
				a[2] = 0 ; 
				a[3] = 0 ;
				a[4] = 0 ;
			}
			else
			{
				gsl_vector *ResExp = gsl_vector_calloc( GS->size ) ;
				gsl_vector *ResExp2 ;
				PDdotGsl( POpSz[L-nstr-2] , GS , Res , 2 ) ;
				PDdotGsl( POpSz[L-nstr-1] , Res, ResExp, 2 ) ;
				gsl_vector_free( Res ) ;
				Res = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpSz[L-nstr-1] , ResExp, Res, 0 ) ;
				PDdotGsl( POpSz[L-nstr-2] , Res , Res2 , 0 ) ;
				gsl_blas_ddot( GS, Res2, &a[0] ) ;

				gsl_vector_free( Res ) ;
				gsl_vector_free( Res2) ;
				Res = gsl_vector_calloc( GS->size ) ;
				Res2= gsl_vector_calloc( GS->size ) ;

				for( int j=1 ; j<5 ; j++ )
				{
					ResExp2 = gsl_vector_calloc( GS->size ) ;
					for( int i=nstr ; i>1 ; i-- )
					{
						PDdotGsl( POpSz[L-i] , ResExp, Res, 2 ) ;
						gsl_vector_add( ResExp2 , Res ) ;
						gsl_vector_free( Res ) ;
						Res = gsl_vector_calloc( GS->size ) ;
					}
					PDdotGsl( POpSz[L-1] , ResExp, Res, 3 ) ;
					gsl_vector_add( ResExp2 , Res ) ;
					gsl_vector_free( Res ) ;
					Res = gsl_vector_calloc( GS->size ) ;
					PDdotGsl( POpSz[L-1] , ResExp, Res, 1 ) ;
					gsl_vector_add( ResExp2 , Res ) ;
					gsl_vector_free( Res ) ;
					Res = gsl_vector_calloc( GS->size ) ;
					for( int i=2 ; i<nstr+1 ; i++ )
					{
						PDdotGsl( POpSz[L-i] , ResExp, Res, 0 ) ;
						gsl_vector_add( ResExp2 , Res ) ;
						gsl_vector_free( Res ) ;
						Res = gsl_vector_calloc( GS->size ) ;
					}
					gsl_vector_memcpy( ResExp , ResExp2 ) ;
					gsl_vector_free( ResExp2 ) ;
					PDdotGsl( POpSz[L-nstr-1] , ResExp, Res, 0 ) ;
					PDdotGsl( POpSz[L-nstr-2] , Res,    Res2, 0 ) ;
					gsl_blas_ddot( GS, Res2, &a[j] ) ;
					a[j] *= pow(PI,j) ;
					for( int k=j ; k>0 ; k-- )
						a[j] /= k+1 ;

					gsl_vector_free( Res ) ;
					gsl_vector_free( Res2) ;
					Res = gsl_vector_calloc( GS->size ) ;
					Res2= gsl_vector_calloc( GS->size ) ;
				}
				gsl_vector_free( ResExp ) ;
			}

			gsl_vector_free(Res ) ;
			gsl_vector_free(Res2) ;
		}
		void POpSzSstrCorrGslSymB( gsl_vector *GS , int l , size_t L , double *a )
		{
			gsl_vector *Res  = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res2 = gsl_vector_calloc( GS->size ) ;
			int nstr = l/2 ;
			if( l < 1 )
			{
				PDdotGsl( POpSz[L-1] , GS,   Res, 3 ) ;
				PDdotGsl( POpSz[L-1] , Res, Res2, 1 ) ;
				gsl_blas_ddot( GS, Res2, &a[0] ) ;

				a[1] = 0 ;
				a[2] = 0 ; 
				a[3] = 0 ;
				a[4] = 0 ;
			}
			else
			{
				gsl_vector *ResExp = gsl_vector_calloc( GS->size ) ;
				gsl_vector *ResExp2 ;

				PDdotGsl( POpSz[L-nstr-1] , GS  , ResExp , 2 ) ;
				for( int i=nstr ; i>1 ; i-- )
				{
					ResExp2 = gsl_vector_calloc( GS->size ) ;
					PDdotGsl( POpSz[L-i] , ResExp,  Res, 2 ) ;
					PDdotGsl( POpSz[L-i] , Res, ResExp2, 2 ) ;
					gsl_vector_scale( ResExp2 , -2 ) ;
					gsl_vector_add( ResExp , ResExp2 ) ;
					gsl_vector_free( Res ) ;
					Res = gsl_vector_calloc( GS->size ) ;
					gsl_vector_free( ResExp2 ) ;
				}
				ResExp2 = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpSz[L-1] , ResExp,  Res, 3 ) ;
				PDdotGsl( POpSz[L-1] , Res, ResExp2, 3 ) ;
				gsl_vector_scale( ResExp2 , -2 ) ;
				gsl_vector_add( ResExp , ResExp2 ) ;
				gsl_vector_free( Res ) ;
				Res = gsl_vector_calloc( GS->size ) ;
				gsl_vector_free( ResExp2 ) ;
				ResExp2 = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpSz[L-1] , ResExp,  Res, 1 ) ;
				PDdotGsl( POpSz[L-1] , Res, ResExp2, 1 ) ;
				gsl_vector_scale( ResExp2 , -2 ) ;
				gsl_vector_add( ResExp , ResExp2 ) ;
				gsl_vector_free( Res ) ;
				Res = gsl_vector_calloc( GS->size ) ;
				gsl_vector_free( ResExp2 ) ;
				for( int i=2 ; i<nstr+1 ; i++ )
				{
					ResExp2 = gsl_vector_calloc( GS->size ) ;
					PDdotGsl( POpSz[L-i] , ResExp,  Res, 0 ) ;
					PDdotGsl( POpSz[L-i] , Res, ResExp2, 0 ) ;
					gsl_vector_scale( ResExp2 , -2 ) ;
					gsl_vector_add( ResExp , ResExp2 ) ;
					gsl_vector_free( Res ) ;
					Res = gsl_vector_calloc( GS->size ) ;
					gsl_vector_free( ResExp2 ) ;
				}
				PDdotGsl( POpSz[L-nstr-1] , ResExp , Res , 0 ) ;
				gsl_blas_ddot( GS, Res, &a[0] ) ;

				gsl_vector_free( ResExp ) ;
			}

			gsl_vector_free(Res ) ;
			gsl_vector_free(Res2) ;
		}
		void POpSxSstrCorrGslAsymD( gsl_vector *GS , int l , size_t L , double *a )
		{
			gsl_vector *Res  = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res2 ;
			int nstr = l/2 + 1 ;
			if( nstr < 2 )	// nstr=1
			{
				PDdotGsl( POpSstrSx[L-2] , GS , Res , 0 ) ;
				Res2 = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpSstrSx[L-1] , Res, Res2, 3 ) ;

				gsl_vector_free( Res ) ;
				gsl_blas_ddot( GS, Res2, &a[0] ) ;
				gsl_vector_free( Res2 ) ;
			}
			else		// nstr=(integer) > 1
			{
				gsl_vector *ResExp ;
				PDdotGsl( POpSstrSx[L-1-nstr] , GS , Res , 0 ) ;

				sparseMat dum ;
				dum.setM(3,3) ; //PD1,PD3
				dum.sDat[0].setDat( 0,2, -1 ) ;
				dum.sDat[1].setDat( 1,1, -1 ) ;
				dum.sDat[2].setDat( 2,0, -1 ) ;

				Res2 = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( dum , Res , Res2, 1 ) ;
				gsl_vector_free( Res ) ;

				Res = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( dum , Res2, Res , 3 ) ;
				gsl_vector_free( Res2 ) ;

				dum.sMatFree() ;

				ResExp = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpSstrSx[L-nstr] , Res , ResExp , 2 ) ;
				gsl_vector_free( Res ) ;

				gsl_blas_ddot( GS, ResExp, &a[0] ) ;
				gsl_vector_free( ResExp ) ;
			}
		}
		void POpSzSstrCorrGslAsymD( gsl_vector *GS , int l , size_t L , double *a )
		{
			gsl_vector *Res  = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res2 ;
			int nstr = l/2 + 1 ;
			if( nstr < 2 ) // nstr=1 
			{
				PDdotGsl( POpSstrSz[L-2] , GS , Res , 0 ) ;
				Res2 = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpSstrSz[L-1] , Res, Res2, 3 ) ;

				gsl_vector_free( Res ) ;
				gsl_blas_ddot( GS, Res2, &a[0] ) ;
				gsl_vector_free( Res2 ) ;
			}
			else		// nstr=integer > 1
			{
				gsl_vector *ResExp ;
				PDdotGsl( POpSstrSz[L-1-nstr] , GS , Res , 0 ) ;

				sparseMat dum ;
				dum.setM(3,3) ; //PD1,PD3
				dum.sDat[0].setDat( 0,0, -1 ) ;
				dum.sDat[1].setDat( 1,1,  1 ) ;
				dum.sDat[2].setDat( 2,2, -1 ) ;

				Res2 = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( dum , Res , Res2, 1 ) ;
				gsl_vector_free( Res ) ;

				Res = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( dum , Res2, Res , 3 ) ;
				gsl_vector_free( Res2 ) ;

				dum.sMatFree() ;

				ResExp = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpSstrSz[L-nstr] , Res , ResExp , 2 ) ;
				gsl_vector_free( Res ) ;

				gsl_blas_ddot( GS, ResExp, &a[0] ) ;
				gsl_vector_free( ResExp ) ;
			}
		}
		void POpSxSstrCorrGslSymD( gsl_vector *GS , int l , size_t L , double *a )
		{
			gsl_vector *Res  = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res2 ;
			int nstr = l/2 ;
			if( nstr < 1 )	// nstr=0
			{
				PDdotGsl( POpSstrSx[L-1] , GS , Res , 1 ) ;
				Res2 = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpSstrSx[L-1] , Res, Res2, 3 ) ;

				gsl_vector_free( Res ) ;
				gsl_blas_ddot( GS, Res2, &a[0] ) ;
				gsl_vector_free( Res2 ) ;
			}
			else		// nstr=(positive integer)
			{
				gsl_vector *ResExp ;
				PDdotGsl( POpSstrSx[L-1-nstr] , GS , Res , 0 ) ;

				sparseMat dum ;
				dum.setM(3,3) ; //PD1,PD3
				dum.sDat[0].setDat( 0,2, -1 ) ;
				dum.sDat[1].setDat( 1,1, -1 ) ;
				dum.sDat[2].setDat( 2,0, -1 ) ;

				Res2 = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( dum , Res , Res2, 1 ) ;
				gsl_vector_free( Res ) ;

				Res = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( dum , Res2, Res , 3 ) ;
				gsl_vector_free( Res2 ) ;

				dum.sMatFree() ;

				ResExp = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpSstrSx[L-1-nstr] , Res , ResExp , 2 ) ;
				gsl_vector_free( Res ) ;

				gsl_blas_ddot( GS, ResExp, &a[0] ) ;
				gsl_vector_free( ResExp ) ;
			}
		}
		void POpSzSstrCorrGslSymD( gsl_vector *GS , int l , size_t L , double *a )
		{
			gsl_vector *Res  = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res2 ;
			int nstr = l/2 ;
			if( nstr < 1 )
			{
				PDdotGsl( POpSstrSz[L-1] , GS , Res , 1 ) ;
				Res2 = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpSstrSz[L-1] , Res, Res2, 3 ) ;

				gsl_vector_free( Res ) ;
				gsl_blas_ddot( GS, Res2, &a[0] ) ;
				gsl_vector_free( Res2 ) ;
			}
			else
			{
				gsl_vector *ResExp ;
				PDdotGsl( POpSstrSz[L-1-nstr] , GS , Res , 0 ) ;

				sparseMat dum ;
				dum.setM(3,3) ; //PD1,PD3
				dum.sDat[0].setDat( 0,0, -1 ) ;
				dum.sDat[1].setDat( 1,1,  1 ) ;
				dum.sDat[2].setDat( 2,2, -1 ) ;

				Res2 = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( dum , Res , Res2, 1 ) ;
				gsl_vector_free( Res ) ;

				Res = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( dum , Res2, Res , 3 ) ;
				gsl_vector_free( Res2 ) ;

				dum.sMatFree() ;

				ResExp = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpSstrSz[L-1-nstr] , Res , ResExp , 2 ) ;
				gsl_vector_free( Res ) ;

				gsl_blas_ddot( GS, ResExp, &a[0] ) ;
				gsl_vector_free( ResExp ) ;
			}
		}
		void POpSzStrCorrGslSymB( gsl_vector *GS , int l , size_t L , double *a )
		{
			gsl_vector *Res  = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res2 = gsl_vector_calloc( GS->size ) ;
			int nstr = l/2 ;
			if( l < 1 )
			{
				PDdotGsl( POpSz[L-2] , GS , Res , 2 ) ;
				PDdotGsl( POpSz[L-1] , Res, Res2, 3 ) ;
				gsl_vector_free( Res ) ;
				Res = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpSz[L-1] , Res2, Res , 1 ) ;
				gsl_vector_free( Res2 ) ;
				Res2 = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpSz[L-2] , Res, Res2, 0 ) ;
				gsl_blas_ddot( GS, Res2, &a[0] ) ;
				a[1] = 0 ;
				a[2] = 0 ; 
				a[3] = 0 ;
				a[4] = 0 ;
			}
			else
			{
				gsl_vector *ResExp = gsl_vector_calloc( GS->size ) ;
				gsl_vector *ResExp2 ;
				PDdotGsl( POpSz[L-nstr-2] , GS , Res , 2 ) ;
				PDdotGsl( POpSz[L-nstr-1] , Res, ResExp, 2 ) ;

				gsl_vector_free( Res ) ;
				Res = gsl_vector_calloc( GS->size ) ;
				ResExp2 = gsl_vector_calloc( GS->size ) ;
				for( int i=nstr ; i>1 ; i-- )
				{
					PDdotGsl( POpSz[L-i] , ResExp,  Res, 2 ) ;
					PDdotGsl( POpSz[L-i] , Res, ResExp2, 2 ) ;
					gsl_vector_scale( ResExp2 , -2 ) ;
					gsl_vector_add( ResExp , ResExp2 ) ;
					gsl_vector_free( Res ) ;
					Res = gsl_vector_calloc( GS->size ) ;
					gsl_vector_free( ResExp2 ) ;
					ResExp2 = gsl_vector_calloc( GS->size ) ;
				}
				PDdotGsl( POpSz[L-1] , ResExp,  Res, 3 ) ;
				PDdotGsl( POpSz[L-1] , Res, ResExp2, 3 ) ;
				gsl_vector_scale( ResExp2 , -2 ) ;
				gsl_vector_add( ResExp , ResExp2 ) ;
				gsl_vector_free( Res ) ;
				Res = gsl_vector_calloc( GS->size ) ;
				gsl_vector_free( ResExp2 ) ;
				ResExp2 = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpSz[L-1] , ResExp,  Res, 1 ) ;
				PDdotGsl( POpSz[L-1] , Res, ResExp2, 1 ) ;
				gsl_vector_scale( ResExp2 , -2 ) ;
				gsl_vector_add( ResExp , ResExp2 ) ;
				gsl_vector_free( Res ) ;
				Res = gsl_vector_calloc( GS->size ) ;
				gsl_vector_free( ResExp2 ) ;
				ResExp2 = gsl_vector_calloc( GS->size ) ;
				for( int i=2 ; i<nstr+1 ; i++ )
				{
					PDdotGsl( POpSz[L-i] , ResExp,  Res, 0 ) ;
					PDdotGsl( POpSz[L-i] , Res, ResExp2, 0 ) ;
					gsl_vector_scale( ResExp2 , -2 ) ;
					gsl_vector_add( ResExp , ResExp2 ) ;
					gsl_vector_free( Res ) ;
					Res = gsl_vector_calloc( GS->size ) ;
					gsl_vector_free( ResExp2 ) ;
					ResExp2 = gsl_vector_calloc( GS->size ) ;
				}
				gsl_vector_free( ResExp2 ) ;

				gsl_vector_free( Res ) ;
				Res = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpSz[L-nstr-1] , ResExp, Res  , 0 ) ;
				PDdotGsl( POpSz[L-nstr-2] , Res , ResExp , 0 ) ;
				gsl_blas_ddot( GS, ResExp, &a[0] ) ;

				gsl_vector_free( ResExp ) ;
			}

			gsl_vector_free(Res ) ;
			gsl_vector_free(Res2) ;
		}
		void POpSzStrCorrGslSymD( gsl_vector *GS , int l , size_t L , double *a )
		{
			gsl_vector *Res  = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res2 ;
			int nstr = l/2 ;
			if( nstr < 1 )
			{
				PDdotGsl( POpStrSz[L-2] , GS , Res , 4 ) ;
				Res2 = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpStrSz[L-2] , Res, Res2, 5 ) ;

				gsl_vector_free( Res ) ;
				gsl_blas_ddot( GS, Res2, &a[0] ) ;
				gsl_vector_free( Res2 ) ;
			}
			else
			{
				gsl_vector *ResExp ;
				PDdotGsl( POpStrSz[L-2-nstr] , GS , Res , 0 ) ;

				sparseMat dum ;
				dum.setM(3,3) ; //PD1,PD3
				dum.sDat[0].setDat( 0,0, -1 ) ;
				dum.sDat[1].setDat( 1,1,  1 ) ;
				dum.sDat[2].setDat( 2,2, -1 ) ;

				Res2 = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( dum , Res , Res2, 1 ) ;
				gsl_vector_free( Res ) ;

				Res = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( dum , Res2, Res , 3 ) ;
				gsl_vector_free( Res2 ) ;

				dum.sMatFree() ;

				ResExp = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpStrSz[L-2-nstr] , Res , ResExp , 2 ) ;
				gsl_vector_free( Res ) ;

				gsl_blas_ddot( GS, ResExp, &a[0] ) ;
				gsl_vector_free( ResExp ) ;
			}
		}
		void POpSzStrCorrGslAsymD( gsl_vector *GS , int l , size_t L , double *a )
		{
			gsl_vector *Res  = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res2 ;
			int nstr = l/2 + 1 ;
			if( nstr < 2 )	// nstr=1
			{
				PDdotGsl( POpStrSz[L-3] , GS , Res , 0 ) ;

				sparseMat dum ;
				dum.setM(3,3) ; //PD1,PD3
				dum.sDat[0].setDat( 0,0, -1 ) ;
				dum.sDat[1].setDat( 1,1,  1 ) ;
				dum.sDat[2].setDat( 2,2, -1 ) ;

				Res2 = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( dum , Res , Res2, 1 ) ;
				gsl_vector_free( Res ) ;
				dum.sMatFree() ;

				Res = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpStrSz[L-2] , Res2, Res, 5 ) ;
				gsl_vector_free( Res2 ) ;

				gsl_blas_ddot( GS, Res, &a[0] ) ;
				gsl_vector_free( Res ) ;
			}
			else
			{
				gsl_vector *ResExp ;
				PDdotGsl( POpStrSz[L-2-nstr] , GS , Res , 0 ) ;

				sparseMat dum ;
				dum.setM(3,3) ; //PD1,PD3
				dum.sDat[0].setDat( 0,0, -1 ) ;
				dum.sDat[1].setDat( 1,1,  1 ) ;
				dum.sDat[2].setDat( 2,2, -1 ) ;

				Res2 = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( dum , Res , Res2, 1 ) ;
				gsl_vector_free( Res ) ;

				Res = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( dum , Res2, Res , 3 ) ;
				gsl_vector_free( Res2 ) ;

				dum.sMatFree() ;

				ResExp = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpStrSz[L-1-nstr] , Res , ResExp , 2 ) ;
				gsl_vector_free( Res ) ;

				gsl_blas_ddot( GS, ResExp, &a[0] ) ;
				gsl_vector_free( ResExp ) ;
			}
		}
		void POpSxStrCorrGslSymD( gsl_vector *GS , int l , size_t L , double *a )
		{
			gsl_vector *Res  = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res2 ;
			int nstr = l/2 ;
			if( nstr < 1 )
			{
				PDdotGsl( POpStrSx[L-2] , GS , Res , 4 ) ;
				Res2 = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpStrSx[L-2] , Res, Res2, 5 ) ;

				gsl_vector_free( Res ) ;
				gsl_blas_ddot( GS, Res2, &a[0] ) ;
				gsl_vector_free( Res2 ) ;
			}
			else
			{
				gsl_vector *ResExp ;
				PDdotGsl( POpStrSx[L-2-nstr] , GS , Res , 0 ) ;

				sparseMat dum ;
				dum.setM(3,3) ; //PD1,PD3
				dum.sDat[0].setDat( 0,2, -1 ) ;
				dum.sDat[1].setDat( 1,1, -1 ) ;
				dum.sDat[2].setDat( 2,0, -1 ) ;

				Res2 = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( dum , Res , Res2, 1 ) ;
				gsl_vector_free( Res ) ;

				Res = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( dum , Res2, Res , 3 ) ;
				gsl_vector_free( Res2 ) ;

				dum.sMatFree() ;

				ResExp = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpStrSx[L-2-nstr] , Res , ResExp , 2 ) ;
				gsl_vector_free( Res ) ;

				gsl_blas_ddot( GS, ResExp, &a[0] ) ;
				gsl_vector_free( ResExp ) ;
			}
		}
		void POpSxStrCorrGslAsymD( gsl_vector *GS , int l , size_t L , double *a )
		{
			gsl_vector *Res  = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res2 ;
			int nstr = l/2 + 1;
			if( nstr < 2 )
			{
				PDdotGsl( POpStrSx[L-3] , GS , Res , 0 ) ;

				sparseMat dum ;
				dum.setM(3,3) ; //PD1,PD3
				dum.sDat[0].setDat( 0,2, -1 ) ;
				dum.sDat[1].setDat( 1,1, -1 ) ;
				dum.sDat[2].setDat( 2,0, -1 ) ;

				Res2 = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( dum , Res , Res2, 1 ) ;
				gsl_vector_free( Res ) ;
				dum.sMatFree() ;

				Res = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpStrSx[L-2] , Res2, Res, 5 ) ;
				gsl_vector_free( Res2 ) ;

				gsl_blas_ddot( GS, Res, &a[0] ) ;
				gsl_vector_free( Res ) ;
			}
			else
			{
				gsl_vector *ResExp ;
				PDdotGsl( POpStrSx[L-2-nstr] , GS , Res , 0 ) ;

				sparseMat dum ;
				dum.setM(3,3) ; //PD1,PD3
				dum.sDat[0].setDat( 0,2, -1 ) ;
				dum.sDat[1].setDat( 1,1, -1 ) ;
				dum.sDat[2].setDat( 2,0, -1 ) ;

				Res2 = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( dum , Res , Res2, 1 ) ;
				gsl_vector_free( Res ) ;

				Res = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( dum , Res2, Res , 3 ) ;
				gsl_vector_free( Res2 ) ;

				dum.sMatFree() ;

				ResExp = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpStrSx[L-1-nstr] , Res , ResExp , 2 ) ;
				gsl_vector_free( Res ) ;

				gsl_blas_ddot( GS, ResExp, &a[0] ) ;
				gsl_vector_free( ResExp ) ;
			}
		}
		void POpSzStrCorrGslAsymB( gsl_vector *GS , int l , size_t L , double *a )
		{
			gsl_vector *Res  = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res2 = gsl_vector_calloc( GS->size ) ;
			int nstr = l/2 ;
			if( l < 1 )
			{
				PDdotGsl( POpSz[L-2] , GS , Res , 2 ) ;
				PDdotGsl( POpSz[L-1] , Res, Res2, 3 ) ;
				gsl_vector_free( Res ) ;
				Res = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpSz[L-1] , Res2, Res , 1 ) ;
				gsl_vector_free( Res2 ) ;
				Res2 = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpSz[L-2] , Res, Res2, 0 ) ;
				gsl_blas_ddot( GS, Res2, &a[0] ) ;
				a[1] = 0 ;
				a[2] = 0 ; 
				a[3] = 0 ;
				a[4] = 0 ;
			}
			else
			{
				gsl_vector *ResExp = gsl_vector_calloc( GS->size ) ;
				gsl_vector *ResExp2 ;
				PDdotGsl( POpSz[L-nstr-2] , GS , Res , 2 ) ;
				PDdotGsl( POpSz[L-nstr-1] , Res, Res2, 2 ) ;
				gsl_vector_free( Res ) ;
				Res = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpSz[L-nstr-1] , Res2, Res  , 0 ) ;
				PDdotGsl( POpSz[L-nstr-2] , Res , ResExp , 0 ) ;
				gsl_blas_ddot( GS, ResExp, &a[0] ) ;

				gsl_vector_free( Res ) ;
				gsl_vector_free( Res2) ;
				Res = gsl_vector_calloc( GS->size ) ;
				Res2= gsl_vector_calloc( GS->size ) ;

				ResExp2 = gsl_vector_calloc( GS->size ) ;
				for( int i=nstr ; i>1 ; i-- )
				{
					PDdotGsl( POpSz[L-i] , ResExp,  Res, 2 ) ;
					PDdotGsl( POpSz[L-i] , Res, ResExp2, 2 ) ;
					gsl_vector_scale( ResExp2 , -2 ) ;
					gsl_vector_add( ResExp , ResExp2 ) ;
					gsl_vector_free( Res ) ;
					Res = gsl_vector_calloc( GS->size ) ;
					gsl_vector_free( ResExp2 ) ;
					ResExp2 = gsl_vector_calloc( GS->size ) ;
				}
				PDdotGsl( POpSz[L-1] , ResExp,  Res, 3 ) ;
				PDdotGsl( POpSz[L-1] , Res, ResExp2, 3 ) ;
				gsl_vector_scale( ResExp2 , -2 ) ;
				gsl_vector_add( ResExp , ResExp2 ) ;
				gsl_vector_free( Res ) ;
				Res = gsl_vector_calloc( GS->size ) ;
				gsl_vector_free( ResExp2 ) ;
				ResExp2 = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpSz[L-1] , ResExp,  Res, 1 ) ;
				PDdotGsl( POpSz[L-1] , Res, ResExp2, 1 ) ;
				gsl_vector_scale( ResExp2 , -2 ) ;
				gsl_vector_add( ResExp , ResExp2 ) ;
				gsl_vector_free( Res ) ;
				Res = gsl_vector_calloc( GS->size ) ;
				gsl_vector_free( ResExp2 ) ;
				ResExp2 = gsl_vector_calloc( GS->size ) ;
				for( int i=2 ; i<nstr+1 ; i++ )
				{
					PDdotGsl( POpSz[L-i] , ResExp,  Res, 0 ) ;
					PDdotGsl( POpSz[L-i] , Res, ResExp2, 0 ) ;
					gsl_vector_scale( ResExp2 , -2 ) ;
					gsl_vector_add( ResExp , ResExp2 ) ;
					gsl_vector_free( Res ) ;
					Res = gsl_vector_calloc( GS->size ) ;
					gsl_vector_free( ResExp2 ) ;
					ResExp2 = gsl_vector_calloc( GS->size ) ;
				}
				gsl_vector_free( ResExp2 ) ;
				gsl_blas_ddot( GS, ResExp, &a[0] ) ;

				gsl_vector_free( ResExp ) ;
			}

			gsl_vector_free(Res ) ;
			gsl_vector_free(Res2) ;
		}
		void POpSzStrCorrGslAsym( gsl_vector *GS , int l , size_t L , double *a )
		{
			gsl_vector *Res  = gsl_vector_calloc( GS->size ) ;
			gsl_vector *Res2 = gsl_vector_calloc( GS->size ) ;
			int nstr = (l+1)/2 ;
			if( nstr < 2 ) 
			{
				gsl_vector *ResExp = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpSz[L-nstr-1] , GS , Res , 2 ) ;
				PDdotGsl( POpSz[L-nstr  ] , Res, ResExp, 3 ) ;
				gsl_vector_free( Res ) ;
				Res = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpSz[L-nstr-1] , ResExp, Res, 0 ) ;
				PDdotGsl( POpSz[L-nstr-2] , Res , Res2 , 0 ) ;
				gsl_blas_ddot( GS, Res2, &a[0] ) ;

				gsl_vector_free( Res ) ;
				gsl_vector_free( Res2) ;
				Res = gsl_vector_calloc( GS->size ) ;
				Res2= gsl_vector_calloc( GS->size ) ;

				for( int j=1 ; j<5 ; j++ )
				{
					PDdotGsl( POpSz[L-1] , ResExp, Res, 1 ) ;
					gsl_vector_memcpy( ResExp , Res ) ;
					gsl_vector_free( Res ) ;
					Res = gsl_vector_calloc( GS->size ) ;
					PDdotGsl( POpSz[L-nstr-1] , ResExp, Res, 0 ) ;
					PDdotGsl( POpSz[L-nstr-2] , Res,    Res2, 0 ) ;
					gsl_blas_ddot( GS, Res2, &a[j] ) ;
					a[j] *= pow(PI,j) ;
					for( int k=j ; k>0 ; k-- )
						a[j] /= k+1 ;

					gsl_vector_free( Res ) ;
					gsl_vector_free( Res2) ;
					Res = gsl_vector_calloc( GS->size ) ;
					Res2= gsl_vector_calloc( GS->size ) ;
				}
				gsl_vector_free( ResExp ) ;
			}
			else
			{
				gsl_vector *ResExp = gsl_vector_calloc( GS->size ) ;
				gsl_vector *ResExp2 ;
				PDdotGsl( POpSz[L-nstr-1] , GS , Res , 2 ) ;
				PDdotGsl( POpSz[L-nstr  ] , Res, ResExp, 2 ) ;
				gsl_vector_free( Res ) ;
				Res = gsl_vector_calloc( GS->size ) ;
				PDdotGsl( POpSz[L-nstr-1] , ResExp, Res, 0 ) ;
				PDdotGsl( POpSz[L-nstr-2] , Res , Res2 , 0 ) ;
				gsl_blas_ddot( GS, Res2, &a[0] ) ;

				gsl_vector_free( Res ) ;
				gsl_vector_free( Res2) ;
				Res = gsl_vector_calloc( GS->size ) ;
				Res2= gsl_vector_calloc( GS->size ) ;

				for( int j=1 ; j<5 ; j++ )
				{
					ResExp2 = gsl_vector_calloc( GS->size ) ;
					for( int i=nstr-1 ; i>1 ; i-- )
					{
						PDdotGsl( POpSz[L-i] , ResExp, Res, 2 ) ;
						gsl_vector_add( ResExp2 , Res ) ;
						gsl_vector_free( Res ) ;
						Res = gsl_vector_calloc( GS->size ) ;
					}
					PDdotGsl( POpSz[L-1] , ResExp, Res, 3 ) ;
					gsl_vector_add( ResExp2 , Res ) ;
					gsl_vector_free( Res ) ;
					Res = gsl_vector_calloc( GS->size ) ;
					PDdotGsl( POpSz[L-1] , ResExp, Res, 1 ) ;
					gsl_vector_add( ResExp2 , Res ) ;
					gsl_vector_free( Res ) ;
					Res = gsl_vector_calloc( GS->size ) ;
					for( int i=2 ; i<nstr+1 ; i++ )
					{
						PDdotGsl( POpSz[L-i] , ResExp, Res, 0 ) ;
						gsl_vector_add( ResExp2 , Res ) ;
						gsl_vector_free( Res ) ;
						Res = gsl_vector_calloc( GS->size ) ;
					}
					gsl_vector_memcpy( ResExp , ResExp2 ) ;
					gsl_vector_free( ResExp2 ) ;
					PDdotGsl( POpSz[L-nstr-1] , ResExp, Res, 0 ) ;
					PDdotGsl( POpSz[L-nstr-2] , Res,    Res2, 0 ) ;
					gsl_blas_ddot( GS, Res2, &a[j] ) ;
					a[j] *= pow(PI,j) ;
					for( int k=j ; k>0 ; k-- )
						a[j] /= k+1 ;

					gsl_vector_free( Res ) ;
					gsl_vector_free( Res2) ;
					Res = gsl_vector_calloc( GS->size ) ;
					Res2= gsl_vector_calloc( GS->size ) ;
				}
				gsl_vector_free( ResExp ) ;
			}

			gsl_vector_free(Res ) ;
			gsl_vector_free(Res2) ;
		}
} ;
class Ldata 
{
	public :
		double E ;
		gsl_vector *q ;
		short int def ;

		Ldata(void) {}
		Ldata( double ee, gsl_vector *qq , size_t dim )
		{
			E = ee ;
			q = gsl_vector_alloc( dim ) ;
			gsl_vector_memcpy( q, qq ) ;
			def = 0 ;
		}
		void comp( double ee, gsl_vector *qq ) 
		{
			if( ee < E ) {
				E = ee ;
				gsl_vector_memcpy( q, qq ) ;
				def = 0 ;
			}
			else 
				def++ ;
		}
		void ret( double ee, gsl_vector *qq )
		{
			ee = E ;
			gsl_vector_memcpy( qq, q ) ;
			gsl_vector_free( q ) ;
		}
} ;
#endif 
