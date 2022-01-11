//#include "class_sparse_J2NegE_omp.cpp"

class sup_sub_block {
	public :
		size_t  num_site ,
			num_block_state ,
			num_ssite_state ,
			num_sys_state ,
			num_trunc_state ;
		sparseMat sysHamil ,
			  sysBlockHamil ,
			  totHamil ;
		sparseMatOpform opfTotHamil ;
		sparseMatOpformObChiS opfOb ;
		gsl_matrix *red_dm	;
		gsl_vector *dum_gs ;
		gsl_vector * red_dm_eval ;
		gsl_matrix * red_dm_evec ;
		double tot_eval_gs ,
		       tot_eval_gs_prev ,
		       sys_eval_gs ,
		       sum_weight ,
		       deltaE ,
		       totSz ,
		       totStSz ,
		       totStSzNedge ;
		sopComp sysBlockSop ,
			sysBlockSop2 ,
			ssiteSop ;
		size_t lanczosIter ; 
		double Jz ,
		       J2 ;
		double nDecDigit ,
		       EE ;
		double *stotcomp ;
		sup_sub_block( void ) {		// Constructor
			num_site = 2 ;
			num_block_state = DIMH ;
			num_ssite_state = DIM ;
			num_sys_state = num_block_state * num_ssite_state ;
			lanczosIter = 0 ;
		}
		sup_sub_block( size_t site, size_t block, size_t ssite ) // Constructor2
			: num_site(site), num_block_state(block), num_ssite_state(ssite) {
			num_sys_state = num_block_state * num_ssite_state ;
		}
		void set_init_num( size_t site, size_t blockSt, size_t ssiteSt ) {
			num_site = site ;
			num_block_state = blockSt ;
			num_ssite_state = ssiteSt ;
			num_sys_state = num_block_state * num_ssite_state ;
		}
		void setJz( double JJz )
		{
			Jz = JJz ;
		}
		void setJ2( double JJ2 )
		{
			J2 = JJ2 ;
		}
		void setInitSop( sopSpinHalfCoup spSopBlock , sopSpinOne spSopSsite , double JJend, double JJz , double JJ2 ) {
			if ( num_site != 2 ) 
				printf("ERROR :: In 'setInitSop(..)', initial condition of L=2 is not matched.\n" ) ;
			Jz = JJz ;
			J2 = JJ2 ;
			sysBlockSop.allocCpy( spSopBlock ) ;
			ssiteSop.allocCpy   ( spSopSsite ) ;
			sparseMat dumB ,
				  dumC ;
			sysHamil.prodSop( sysBlockSop.Sz , ssiteSop.Sz , Jz * JJend ) ;
			dumB.prodSop( sysBlockSop.Sp , ssiteSop.Sm , 0.5*JJend) ;
			dumC.prodSop( sysBlockSop.Sm , ssiteSop.Sp , 0.5*JJend) ;

			sysHamil.addSop3( dumB ) ;
			sysHamil.addSop3( dumC ) ;
			dumB.sMatFree() ;
			dumC.sMatFree() ;
			tot_eval_gs_prev = 0 ; 

			opfOb.allocPOp( num_site ) ;
			opfOb.setDim( num_block_state, num_ssite_state ) ;

			// need only if calcPSz and calcPSzCorr
			opfOb.POpSz[1] = ssiteSop.Sz ;
			opfOb.POpSz[0] = sysBlockSop.Sz ;

			// need only if calcPLocalSS (optional)
			//opfOb.POpSS[0] = sysHamil ; 

			sysBlockSop2.Sz.setM( num_block_state, 0 ) ;
			sysBlockSop2.Sp.setM( num_block_state, 0 ) ;
			sysBlockSop2.Sm.setM( num_block_state, 0 ) ;

		} //setInitSop()
		void opfSopLanczosDiagSymSupblock( size_t num_tr , char * ooutfdir , char * ooutfname) {
			opfTotHamil.alloc() ;
			opfTotHamil.setDim( num_block_state , num_ssite_state ) ;
			opfTotHamil.setJz( Jz ) ;
			opfTotHamil.setJ2( J2 ) ;
			opfTotHamil.set1( sysHamil ) ;
			opfTotHamil.set2( sysHamil ) ;
			opfTotHamil.setI( 0 , ssiteSop.Sz , ssiteSop.Sz ) ;
			opfTotHamil.setI( 1 , ssiteSop.Sp , ssiteSop.Sm ) ;
			opfTotHamil.setI( 2 , ssiteSop.Sm , ssiteSop.Sp ) ;
			opfTotHamil.setI2( sysBlockSop2.Sz , sysBlockSop2.Sp , sysBlockSop2.Sm ) ;

			size_t dim_mat = num_sys_state * num_sys_state ;

			dum_gs     = gsl_vector_alloc ( dim_mat ) ;

			opfSopIterModLanczosRNlp( opfTotHamil, dum_gs , tot_eval_gs , dim_mat , lanczosIter , deltaE , ooutfdir, num_site , tot_eval_gs_prev , nDecDigit ) ;
			//tot_eval_gs = tot_eval_gs / num_site / 2. ;

			gsl_matrix_view dum_mps = gsl_matrix_view_vector( dum_gs, num_sys_state, num_sys_state ) ;
			red_dm = gsl_matrix_calloc( num_sys_state, num_sys_state ) ;
			gsl_blas_dgemm( CblasTrans , CblasNoTrans ,
					1. , &dum_mps.matrix , &dum_mps.matrix ,
					0. , red_dm ) ;

			// Diagonalize the reduced density matrix(RDM) and sort by descending order of weight.
			gsl_eigen_symmv_workspace * w ;
			w = gsl_eigen_symmv_alloc( num_sys_state ) ;
			red_dm_eval = gsl_vector_alloc( num_sys_state ) ;
			red_dm_evec = gsl_matrix_alloc( num_sys_state , num_sys_state ) ;
			gsl_eigen_symmv( red_dm, red_dm_eval, red_dm_evec, w ) ;
			gsl_eigen_symmv_sort( red_dm_eval , red_dm_evec, GSL_EIGEN_SORT_VAL_DESC ) ;
			gsl_eigen_symmv_free( w ) ;

			// Set number of states going to be truncated.
			if( num_sys_state < num_tr ) 
				num_trunc_state = num_sys_state ;
			else
				num_trunc_state = num_tr ;

			// Before activating calc() options, must CHECK the update variables.
			calcPSzWrite( ooutfdir ) ;
			calcPSzCorrWrite( ooutfdir ) ;
			calcEE( ooutfdir ) ;
			//calcPLocalSSWrite( ooutfdir ) ;
			write( ooutfname ) ;
			saveNDelSparse( ooutfdir ) ;

			cout <<"Lanczos[L="<<num_site<<"]::completed\n" << flush ;

		} //opfSopLanczosDiagSymSupblock()
		void freeInter() 
		{
			opfTotHamil.opfFree() ;
			//gsl_vector_free( dum_gs ) ;
			gsl_matrix_free( red_dm ) ;

		}
		void updateSop( sup_sub_block prev , double JJz , double JJ2 ) {
			Jz = JJz ;
			J2 = JJ2 ;
			//gsl_vector_free( prev.dum_gs ) ;

			// Set Indicators.
			num_site = 1 + prev.num_site ;
			num_block_state = prev.num_trunc_state ;
			num_ssite_state = prev.num_ssite_state ;
			num_sys_state = num_block_state * num_ssite_state ;

			cout <<"update::L_sys to "<< num_site <<';' << flush ;
			sysBlockSop.Sz = prev.projectSop2aNz( prev.ssiteSop.Sz ) ;	//ssite to block
			sysBlockSop.Sp = prev.projectSop2aNz( prev.ssiteSop.Sp ) ;	//ssite to block
			sysBlockSop.Sm = prev.projectSop2aNz( prev.ssiteSop.Sm ) ;	//ssite to block
			sysBlockSop2.Sz = prev.projectSopSMatNz( prev.sysBlockSop.Sz ) ;	//prev.block to block
			sysBlockSop2.Sp = prev.projectSopSMatNz( prev.sysBlockSop.Sp ) ;	//prev.block to block
			sysBlockSop2.Sm = prev.projectSopSMatNz( prev.sysBlockSop.Sm ) ;	//prev.block to block
			sysBlockHamil  = prev.projectSopHamil2Nz( prev.sysHamil ) ;		//prev.sys to block
			ssiteSop = prev.ssiteSop ; 
			prev.sysBlockSop.sopCFree() ;

			tot_eval_gs_prev = prev.tot_eval_gs ; 

			opfOb.allocPOp( num_site ) ;
			opfOb.setDim( num_block_state, num_ssite_state ) ;
			opfOb.POpSz[num_site-1] = ssiteSop.Sz ;
			opfOb.POpSz[num_site-2] = sysBlockSop.Sz ;
			opfOb.POpSz[num_site-3] = sysBlockSop2.Sz ;

			// need only if calcPSz and calcPSzCorr
			for( int i=0 ; i<(int)num_site-3 ; i++ )
			{
				opfOb.POpSz[i] = prev.projectSopSMatNz( prev.opfOb.POpSz[i] ) ;
			}
			prev.sysBlockSop2.sopCFree() ;

			// need only if calcPLocalSS
			//for( int i=0 ; i<(int)num_site-3 ; i++ )
			//{
				//opfOb.POpSS[i] = prev.projectSopSMatNz( prev.opfOb.POpSS[i] ) ;
			//}
			//opfOb.POpSS[num_site-3] = prev.projectSopHamil2Nz( prev.opfOb.POpSS[num_site-3] ) ;

			prev.sysHamil.sMatFree() ;

			sysHamil.prodSopRId( sysBlockHamil , num_ssite_state ) ;
			sysBlockHamil.sMatFree() ;

			// For J1 (need whenever)
			sparseMat dumA ,
				  dumC ,
				  dumD ;
			dumA.prodSop( sysBlockSop.Sz , ssiteSop.Sz , Jz ) ;
			sysHamil.addSop3( dumA ) ; dumA.sMatFree() ;
			dumC.prodSop( sysBlockSop.Sp , ssiteSop.Sm , 0.5 ) ;
			sysHamil.addSop3( dumC ) ; dumC.sMatFree() ;
			dumD.prodSop( sysBlockSop.Sm , ssiteSop.Sp , 0.5 ) ;
			sysHamil.addSop3( dumD ) ; dumD.sMatFree() ;

			// Options for PLocalSS (inserting into the paragraph above)
			//opfOb.POpSS[num_site-2].sMatAllocCpy( dumA );
			//opfOb.POpSS[num_site-2].addSop3( dumA ) ;
			//opfOb.POpSS[num_site-2].addSop3( dumC ) ;
			//opfOb.POpSS[num_site-2].addSop3( dumD ) ;
			//sysHamil.addSop3( opfOb.POpSS[num_site-2] ) ;
			// If PLocalSS considered Use freePOpSS() , if NOT freePOp()
			prev.opfOb.freePOp() ;

			// For J2
			dumA.prodSop( sysBlockSop2.Sz , ssiteSop.Sz , J2*Jz ) ;
			sysHamil.addSop3( dumA ) ; dumA.sMatFree() ;
			dumC.prodSop( sysBlockSop2.Sp , ssiteSop.Sm , J2*0.5 ) ;
			sysHamil.addSop3( dumC ) ; dumC.sMatFree() ;
			dumD.prodSop( sysBlockSop2.Sm , ssiteSop.Sp , J2*0.5 ) ;
			sysHamil.addSop3( dumD ) ; dumD.sMatFree() ;

			gsl_matrix_free( prev.red_dm_evec ) ;
		}
		sparseMat projectSop2( sparseMat SsiteScomp ) {
			gsl_matrix_view trunc_st = gsl_matrix_submatrix( red_dm_evec , 0 , 0 , num_sys_state , num_trunc_state ) ;
			sparseMat dumEnlargeSop ,
				  dumMat ;

			dumEnlargeSop.prodSopLId( SsiteScomp , num_block_state ) ;
			dumMat.setM( num_trunc_state , 0 ) ;
			dumMat.dim = num_trunc_state ;
			dumMat.sizeDat = 0 ;
			dumMat.sDat = (sparseDat*) malloc( sizeof(sparseDat)*num_trunc_state*num_trunc_state ) ;

			for( size_t l=0 ; l<num_trunc_state ; l++ ) {
				for( size_t m=0 ; m<num_trunc_state ; m++ ) {
					double Mlm = 0 ;
					for( size_t ai=0 ; ai<dumEnlargeSop.sizeDat ; ai++ ) {
						double dumval = gsl_matrix_get(&trunc_st.matrix,dumEnlargeSop.sDat[ai].i,l) *
								dumEnlargeSop.sDat[ai].dat *
								gsl_matrix_get(&trunc_st.matrix,dumEnlargeSop.sDat[ai].j,m) ;
						Mlm += dumval ;
					}
					if( fabs(Mlm) > ZEROCUT ) {
						dumMat.sDat[dumMat.sizeDat].i   = l ;
						dumMat.sDat[dumMat.sizeDat].j   = m ;
						dumMat.sDat[dumMat.sizeDat].dat = Mlm ;
						dumMat.sizeDat ++ ;
					}
				}
			}
			dumMat.sDat = (sparseDat*) realloc( dumMat.sDat , sizeof(sparseDat)*dumMat.sizeDat ) ;
			dumEnlargeSop.sMatFree() ;
			return dumMat ;
		}
		sparseMat projectSop2aNz( sparseMat SsiteSMat ) {
			gsl_matrix_view trunc_st = gsl_matrix_submatrix( red_dm_evec , 0 , 0 , num_sys_state , num_trunc_state ) ;
			sparseMat dumMat ;

			//dumMat.setM( num_trunc_state , 0 ) ;
			dumMat.dim = num_trunc_state ;
			dumMat.sizeDat = 0 ;
			dumMat.sDat = (sparseDat*) malloc( sizeof(sparseDat)*num_trunc_state*num_trunc_state ) ;

			for( size_t l=0 ; l<num_trunc_state ; l++ ) {
				for( size_t m=0 ; m<num_trunc_state ; m++ ) {
					double Mlm = 0 ;
					for( size_t bi=0 ; bi<SsiteSMat.sizeDat ; bi++ ) {
						for( size_t ai=0 ; ai<num_block_state ; ai++ ) {
							size_t ii = ai + num_block_state*(SsiteSMat.sDat[bi].i) ;
							size_t jj = ai + num_block_state*(SsiteSMat.sDat[bi].j) ;
							double dumval = gsl_matrix_get(&trunc_st.matrix,ii,l) * SsiteSMat.sDat[bi].dat *
								gsl_matrix_get(&trunc_st.matrix,jj,m) ;
							Mlm += dumval ;
						}
					}
					dumMat.sDat[dumMat.sizeDat].i   = l ;
					dumMat.sDat[dumMat.sizeDat].j   = m ;
					dumMat.sDat[dumMat.sizeDat].dat = Mlm ;
					dumMat.sizeDat ++ ;
				}
			}
			dumMat.sDat = (sparseDat*) realloc( dumMat.sDat , sizeof(sparseDat)*dumMat.sizeDat ) ;
			return dumMat ;
		}
		sparseMat projectSopSz( sparseMat SysMat , int c )
	       	{
			gsl_matrix_view trunc_st = gsl_matrix_submatrix( red_dm_evec , 0 , 0 , num_sys_state , num_trunc_state ) ;
			sparseMat dumMat ;

			dumMat.setM( num_trunc_state , 0 ) ;
			dumMat.dim = num_trunc_state ;
			dumMat.sizeDat = 0 ;
			dumMat.sDat = (sparseDat*) malloc( sizeof(sparseDat)*num_trunc_state*num_trunc_state ) ;

			switch( c )
			{
				case 0 :
					for( size_t l=0 ; l<num_trunc_state ; l++ ) {
						for( size_t m=0 ; m<num_trunc_state ; m++ ) {
							double Mlm = 0 ;
							for( size_t ai=0 ; ai<SysMat.sizeDat ; ai++ ) {
								for( size_t bi=0 ; bi<num_ssite_state ; bi++ ) {
									size_t ii = (SysMat.sDat[ai].i) + num_block_state*bi ;
									size_t jj = (SysMat.sDat[ai].j) + num_block_state*bi ;
									double dumval = gsl_matrix_get(&trunc_st.matrix,ii,l) * SysMat.sDat[ai].dat *
										gsl_matrix_get(&trunc_st.matrix,jj,m) ;
									Mlm += dumval ;
								}
							}
							if( fabs(Mlm) > ZEROCUT ) {
								dumMat.sDat[dumMat.sizeDat].i   = l ;
								dumMat.sDat[dumMat.sizeDat].j   = m ;
								dumMat.sDat[dumMat.sizeDat].dat = Mlm ;
								dumMat.sizeDat ++ ;
							}
						}
					}
					break ;
				case 1 :
					for( size_t l=0 ; l<num_trunc_state ; l++ ) {
						for( size_t m=0 ; m<num_trunc_state ; m++ ) {
							double Mlm = 0 ;
							for( size_t bi=0 ; bi<SysMat.sizeDat ; bi++ ) {
								for( size_t ai=0 ; ai<num_block_state ; ai++ ) {
									size_t ii = ai + num_ssite_state*(SysMat.sDat[bi].i) ;
									size_t jj = ai + num_ssite_state*(SysMat.sDat[bi].j) ;
									double dumval = gsl_matrix_get(&trunc_st.matrix,ii,l) * SysMat.sDat[bi].dat *
										gsl_matrix_get(&trunc_st.matrix,jj,m) ;
									Mlm += dumval ;
								}
							}
							if( fabs(Mlm) > ZEROCUT ) {
								dumMat.sDat[dumMat.sizeDat].i   = l ;
								dumMat.sDat[dumMat.sizeDat].j   = m ;
								dumMat.sDat[dumMat.sizeDat].dat = Mlm ;
								dumMat.sizeDat ++ ;
							}
						}
					}
					break ;
			}
			dumMat.sDat = (sparseDat*) realloc( dumMat.sDat , sizeof(sparseDat)*dumMat.sizeDat ) ;
			return dumMat ;
		}
		sparseMat projectSopSMatNz( sparseMat SysMat ) 
	       	{
			gsl_matrix_view trunc_st = gsl_matrix_submatrix( red_dm_evec , 0 , 0 , num_sys_state , num_trunc_state ) ;
			sparseMat dumMat ;

			//dumMat.setM( num_trunc_state , 0 ) ;
			dumMat.dim = num_trunc_state ;
			dumMat.sizeDat = 0 ;
			dumMat.sDat = (sparseDat*) malloc( sizeof(sparseDat)*num_trunc_state*num_trunc_state ) ;

			for( size_t l=0 ; l<num_trunc_state ; l++ ) {
				for( size_t m=0 ; m<num_trunc_state ; m++ ) {
					double Mlm = 0 ;
					for( size_t ai=0 ; ai<SysMat.sizeDat ; ai++ ) {
						for( size_t bi=0 ; bi<num_ssite_state ; bi++ ) {
							size_t ii = (SysMat.sDat[ai].i) + num_block_state*bi ;
							size_t jj = (SysMat.sDat[ai].j) + num_block_state*bi ;
							double dumval = gsl_matrix_get(&trunc_st.matrix,ii,l) * SysMat.sDat[ai].dat *
								gsl_matrix_get(&trunc_st.matrix,jj,m) ;
							Mlm += dumval ;
						}
					}
					dumMat.sDat[dumMat.sizeDat].i   = l ;
					dumMat.sDat[dumMat.sizeDat].j   = m ;
					dumMat.sDat[dumMat.sizeDat].dat = Mlm ;
					dumMat.sizeDat ++ ;
				}
			}
			dumMat.sDat = (sparseDat*) realloc( dumMat.sDat , sizeof(sparseDat)*dumMat.sizeDat ) ;
			return dumMat ;
		}
		sparseMat projectSopHamil( sparseMat prevSysHamil ) {
			gsl_matrix_view trunc_st = gsl_matrix_submatrix( red_dm_evec , 0 , 0 , num_sys_state , num_trunc_state ) ;
			sparseMat dumMat ;
			dumMat.setM( num_trunc_state , 0 ) ;

			for( size_t l=0 ; l<num_trunc_state ; l++ ) {
				for( size_t m=0 ; m<num_trunc_state ; m++ ) {
					double Mlm = 0 ;
					for( size_t ai=0 ; ai<prevSysHamil.sizeDat ; ai++ ) {
						double dumval = gsl_matrix_get(&trunc_st.matrix,prevSysHamil.sDat[ai].i,l) *
								prevSysHamil.sDat[ai].dat *
								gsl_matrix_get(&trunc_st.matrix,prevSysHamil.sDat[ai].j,m) ;
						Mlm += dumval ;
					}
					if( fabs(Mlm) > ZEROCUT ) {
						sparseDat n1( l,m, Mlm ) ;
						dumMat.insertDat( n1 ) ;
					}
				}
			}
			return dumMat ;
		}
		sparseMat projectSopHamil2Nz( sparseMat prevSysHamil ) {
			gsl_matrix_view trunc_st = gsl_matrix_submatrix( red_dm_evec , 0 , 0 , num_sys_state , num_trunc_state ) ;
			sparseMat dumMat ;
			dumMat.dim = num_trunc_state ;
			dumMat.sizeDat = 0 ;
			dumMat.sDat = (sparseDat*) malloc( sizeof(sparseDat)*num_trunc_state*num_trunc_state ) ;

			for( size_t l=0 ; l<num_trunc_state ; l++ ) {
				for( size_t m=0 ; m<num_trunc_state ; m++ ) {
					double Mlm = 0 ;
					for( size_t ai=0 ; ai<prevSysHamil.sizeDat ; ai++ ) {
						double dumval = gsl_matrix_get(&trunc_st.matrix,prevSysHamil.sDat[ai].i,l) *
								prevSysHamil.sDat[ai].dat *
								gsl_matrix_get(&trunc_st.matrix,prevSysHamil.sDat[ai].j,m) ;
						Mlm += dumval ;
					}
					dumMat.sDat[dumMat.sizeDat].i   = l ;
					dumMat.sDat[dumMat.sizeDat].j   = m ;
					dumMat.sDat[dumMat.sizeDat].dat = Mlm ;
					dumMat.sizeDat ++ ;
				}
			}
			dumMat.sDat = (sparseDat*) realloc( dumMat.sDat , sizeof(sparseDat)*dumMat.sizeDat ) ;
			return dumMat ;
		}
		void write ( char * ooutfname) {

			//1 char dum[255] ;

			//do{
				//sprintf( ooutfname , "%s/m%zu/Lanczos_m%zu_%d.dat" , dir , mKeep , mKeep , i ) ;
				//sprintf( dum , "test -s %s", ooutfname ) ;
				//i++ ;
			//} while ( system(dum) == 0 ) ;

			ofstream	fw( ooutfname , ios::app ) ;
			//fw << "#Length,m,sys_state,eval,sum_weight" << endl ;

			size_t a=0 ;
			double dum_weight = 0 ;
			for( a=num_trunc_state ; a<(red_dm_eval->size) ; a++ )
				dum_weight += gsl_vector_get( red_dm_eval, a ) ;
			sum_weight = dum_weight ;
			//gsl_vector_free( red_dm_eval ) ;

			fw << scientific ;
			fw << setprecision(2) ;
			fw << setw(4) << num_site << "\t" ;
			fw << setw(4) << num_trunc_state << "\t" << num_block_state << "\t" << num_sys_state << "\t" ;
			fw << setprecision(16) << setw(19) ;
			fw << tot_eval_gs/double(num_site)/2.<< "\t" << sum_weight << "\t" << totSz ;
			fw << setw(5) ;
			fw << "\t" << lanczosIter << "\t" ;
			fw << setprecision(16) << setw(19) << deltaE << "\t" << (tot_eval_gs - tot_eval_gs_prev)/2. << "\t" ;
			fw << setw(5) << setprecision(3) << Jz << "\t" ;
			fw << setprecision(16) << setw(19) << totStSzNedge << "\t" ;
			fw << setw(5) << setprecision(3) << int(nDecDigit) << endl << flush ;
			fw.close() ;


			stringstream ss ;
			ss << ooutfname << '2';
			fw.open( ss.str().c_str() , ios::app ) ;

			fw << setprecision(2) << scientific ;
			fw << setw(4) << num_site << "\t" ;
			fw << setprecision(16) << setw(19) << totStSz << "\t" << EE << "\t"<< stotcomp[0] <<"\t"<< stotcomp[1] <<"\t"<< stotcomp[3] << endl << flush ;
			free(stotcomp) ;


			//1 sprintf( dum , "gnuplot /Storage1/jun/dmrg/Heisenberg/spinOne/plot/gs/energy.gnu" ) ;
			//1 system(dum) ;
		}
		void saveSparse ( char * oooutfdir ) {

			stringstream ss ;
			string dir ;
			ss << oooutfdir << "/L" << num_site << '/' ;
			dir = ss.str() ;
			dir.insert(0, "mkdir -p ") ;
			system( dir.c_str() ) ;

			dir = ss.str() ;
			dir.insert( dir.length() , "num.bin" ) ;

			ofstream	fsave( dir.c_str() , ios_base::binary ) ;
			size_t *snum = new size_t[3] ;
			snum[0] = num_site ;
			snum[1] = num_trunc_state ;
			snum[2] = num_block_state ;
			fsave.write( (char*)snum , sizeof(size_t)*3 ) ;
			fsave.close() ;

			dir = ss.str() ;
			dir.insert( dir.length() , "Sop.bin" ) ;
			fsave.open( dir.c_str() , ios_base::binary ) ;
			fsave.write( (char*)&sysHamil.dim     , sizeof(size_t) ) ;
			fsave.write( (char*)&sysHamil.sizeDat , sizeof(size_t) ) ;
			fsave.write( (char*)sysHamil.sDat     , sizeof(sparseDat)*sysHamil.sizeDat ) ;
			fsave.write( (char*)&tot_eval_gs      , sizeof(double) ) ;
			for( size_t mi=0 ; mi<dum_gs->size ; mi++ ) {
				fsave.write( (char *)gsl_vector_ptr(dum_gs,mi) , sizeof(double) ) ;
			}
			//for( size_t mi=0 ; mi<red_dm_evec->size1 ; mi++ ) {
				//for( size_t mj=0 ; mj<red_dm_evec->size2 ; mj++ ) {
					//fsave.write( (char *)gsl_matrix_ptr(red_dm_evec,mi,mj) , sizeof(double) ) ;
				//}
			//}
			//for( size_t mi=0 ; mi<red_dm_eval->size ; mi++ ) {
				//fsave.write( (char *)gsl_vector_ptr(red_dm_eval,mi) , sizeof(double) ) ;
			//}
			fsave.close() ;
			delete []snum ;
		}
		void saveNDelSparse ( char * oooutfdir ) {

			stringstream ss ;
			string dir ;
			ss << oooutfdir << "/L" << num_site << '/' ;
			dir = ss.str() ;
			dir.insert(0, "mkdir -p ") ;
			system( dir.c_str() ) ;

			dir = ss.str() ;
			dir.insert( dir.length() , "num.bin" ) ;

			ofstream	fsave( dir.c_str() , ios_base::binary ) ;
			size_t *snum = new size_t[3] ;
			snum[0] = num_site ;
			snum[1] = num_trunc_state ;
			snum[2] = num_block_state ;
			fsave.write( (char*)snum , sizeof(size_t)*3 ) ;
			fsave.close() ;

			dir = ss.str() ;
			dir.insert( dir.length() , "Sop.bin" ) ;
			fsave.open( dir.c_str() , ios_base::binary ) ;
			fsave.write( (char*)&sysHamil.dim     , sizeof(size_t) ) ;
			fsave.write( (char*)&sysHamil.sizeDat , sizeof(size_t) ) ;
			fsave.write( (char*)sysHamil.sDat     , sizeof(sparseDat)*sysHamil.sizeDat ) ;
			fsave.write( (char*)&tot_eval_gs      , sizeof(double) ) ;
			for( size_t mi=0 ; mi<dum_gs->size ; mi++ ) {
				fsave.write( (char *)gsl_vector_ptr(dum_gs,mi) , sizeof(double) ) ;
			}
			fsave << flush ;
			//for( size_t mi=0 ; mi<red_dm_evec->size1 ; mi++ ) {
				//for( size_t mj=0 ; mj<red_dm_evec->size2 ; mj++ ) {
					//fsave.write( (char *)gsl_matrix_ptr(red_dm_evec,mi,mj) , sizeof(double) ) ;
				//}
			//}
			//for( size_t mi=0 ; mi<red_dm_eval->size ; mi++ ) {
				//fsave.write( (char *)gsl_vector_ptr(red_dm_eval,mi) , sizeof(double) ) ;
			//}
			fsave.close() ;
			delete []snum ;

			//Delete (num_site-1)-data
			ss.str("") ;
			ss << oooutfdir << "/L" << num_site-1 << '/' ;
			dir = "test -d " ;
			dir += ss.str() ;
			if( system(dir.c_str()) == 0 ) {
				dir = "rm -r " ;
				dir += ss.str() ;
				dir += "Sop.bin" ;
				dir += "&" ;
				system( dir.c_str() ) ;

				dir = "rm -r " ;
				dir += ss.str() ;
				dir += "num.bin" ;
				dir += "&" ;
				system( dir.c_str() ) ;
			}
		}
		void loadSparse ( char * oooutfdir , sopSpinOne spSopSsite ) {
			cout << "Loading::" << oooutfdir << ';' ;
			ssiteSop.allocCpy   ( spSopSsite ) ;


			stringstream ss ;
			string dir ;
			ss << oooutfdir ;
			dir = ss.str() ;
			dir.insert(0, "test -d") ;
			if( system( dir.c_str() ) != 0 ) {
				cout << "ERROR :: NO directory to be loaded" << endl ;
			}

			dir = ss.str() ;
			dir.insert( dir.length() , "num.bin" ) ;

			ifstream	fload( dir.c_str() , ios_base::binary ) ;
			size_t *lnum = new size_t[3] ;
			for( size_t ai=0 ; ai<3 ; ai++ ) {
				fload.read( (char*)&lnum[ai] , sizeof(size_t) ) ;
			}
			num_site        = lnum[0] ;
			num_trunc_state = lnum[1] ;
			num_block_state = lnum[2] ;
			num_sys_state   = num_block_state * DIM ;
			num_ssite_state = DIM ;
			fload.close() ;
			cout << "L=" << num_site << ';' ;

			dir = ss.str() ;
			dir.insert( dir.length() , "Sop.bin" ) ;
			fload.open( dir.c_str() , ios_base::binary ) ;
			fload.read( (char*)&sysHamil.dim     , sizeof(size_t) ) ;
			fload.read( (char*)&sysHamil.sizeDat , sizeof(size_t) ) ;
			sysHamil.sDat = (sparseDat *) malloc( (sizeof(sparseDat)*sysHamil.sizeDat) ) ;
			for( size_t mi=0 ; mi<sysHamil.sizeDat ; mi++ ) {
				fload.read( (char*)&sysHamil.sDat[mi]     , sizeof(sparseDat) ) ;
			}
			fload.read( (char*)&tot_eval_gs      , sizeof(double) ) ;
			dum_gs = gsl_vector_calloc( num_sys_state*num_sys_state ) ;
			for( size_t mi=0 ; mi<dum_gs->size ; mi++ ) {
				fload.read( (char *)gsl_vector_ptr(dum_gs,mi) , sizeof(double) ) ;
			}
			delete []lnum ;

			gsl_matrix_view dum_mps = gsl_matrix_view_vector( dum_gs, num_sys_state, num_sys_state ) ;
			red_dm = gsl_matrix_calloc( num_sys_state, num_sys_state ) ;
			gsl_blas_dgemm( CblasTrans , CblasNoTrans ,
					1. , &dum_mps.matrix , &dum_mps.matrix ,
					0. , red_dm ) ;

			// Diagonalize the reduced density matrix(RDM) and sort by descending order of weight.
			gsl_eigen_symmv_workspace * w ;
			w = gsl_eigen_symmv_alloc( num_sys_state ) ;
			red_dm_eval = gsl_vector_alloc( num_sys_state ) ;
			red_dm_evec = gsl_matrix_alloc( num_sys_state , num_sys_state ) ;
			gsl_eigen_symmv( red_dm, red_dm_eval, red_dm_evec, w ) ;
			gsl_eigen_symmv_sort( red_dm_eval , red_dm_evec, GSL_EIGEN_SORT_VAL_DESC ) ;
			gsl_eigen_symmv_free( w ) ;

			//gsl_vector_free( dum_gs ) ;
			gsl_matrix_free( red_dm ) ;
			cout << "completed\n" ;
		}
		void redEvecWrite( char * oooutfdir ) 
		{

			stringstream ss ;
			string dir ;
			ss << oooutfdir << "/L" << num_site << '/' ;
			dir = ss.str() ;
			dir.insert(0, "mkdir -p ") ;
			system( dir.c_str() ) ;

			dir = ss.str() ;
			dir.insert( dir.length() , "redEvec.nb" ) ;

			ofstream	fsave( dir.c_str() , ios::app ) ;
			fsave << setprecision(16) << scientific ;
			for( unsigned int i=0 ; i<num_sys_state ; i++ ) {
					fsave << "(eval:"<<i<<") " << gsl_vector_get( red_dm_eval, i ) << "\n" ;
			}
			for( unsigned int j=0 ; j<num_sys_state ; j++ ) {
				for( unsigned int i=0 ; i<num_sys_state ; i++ ) {
					fsave << "("<<i<<" , "<<j<<" )= " << gsl_matrix_get( red_dm_evec, i , j ) << "\n" ;
				}
			}
			fsave << endl ;
			fsave.close() ;
		}
		void SzWrite( char * oooutfdir ) 
		{

			stringstream ss ;
			string dir ;
			ss << oooutfdir << "/L" << num_site << '/' ;
			dir = ss.str() ;
			dir.insert(0, "mkdir -p ") ;
			system( dir.c_str() ) ;

			dir = ss.str() ;
			dir.insert( dir.length() , "sopSz.nb" ) ;

			ofstream	fsave( dir.c_str() , ios::app ) ;
			fsave << setprecision(16) << setw(19) << scientific ;
			for( unsigned int i=0 ; i<num_site ; i++ ) {
				fsave << "totS["<<i<<"]:dim="<<opfOb.totS[i].Sz.dim<<",\n" ;
				opfOb.totS[i].Sz.view( fsave ) ;
			}
			fsave << endl ;
			fsave.close() ;
		}
		void gsWrite( char * oooutfdir ) 
		{

			stringstream ss ;
			string dir ;
			ss << oooutfdir << "/L" << num_site << '/' ;
			dir = ss.str() ;
			dir.insert(0, "mkdir -p ") ;
			system( dir.c_str() ) ;

			dir = ss.str() ;
			dir.insert( dir.length() , "gs.nb" ) ;

			ofstream	fsave( dir.c_str() , ios::app ) ;
			fsave << setprecision(16) << setw(19) << scientific ;
			for( unsigned int i=0 ; i<num_sys_state*num_sys_state ; i++ ) {
				fsave << gsl_vector_get( dum_gs , i ) << " ,\n" ;
			}
			fsave << endl ;
			fsave.close() ;
		}
		void calcEE( char * oooutfdir )
		{
			stringstream ss ;
			string dir ;
			ss << oooutfdir << "/L" << num_site << '/' ;
			dir = ss.str() ;
			dir.insert( dir.length() , "calEE2.dat" ) ;

			ofstream	fsave( dir.c_str() , ios::app ) ;
			fsave << setprecision(16) << scientific ;

			double dum = 0 ,
			       dumE= 0 ;
			EE = 0 ;
			for( size_t a=0 ; a<(red_dm_eval->size) ; a++ )
			{
				dum = gsl_vector_get( red_dm_eval, a ) ;
				if( log(dum) == -INFINITY ) {
					fsave << "nan" << "\n" ;
				}       
				else {
					dumE= dum*log(dum) ;
					if( isnan( dumE ) ) {
						fsave << "-nan" << "\n" ;
					}
					else
					{
						EE -= dumE ;
						fsave << -dumE << "\n" ;
					}
				}
			}
			fsave.close() ;
		}
		void calcPSzWrite( char * oooutfdir ) 
		{
			opfOb.POpSzGsl( dum_gs ) ;

			stringstream ss ;
			string dir ;
			ss << oooutfdir << "/L" << num_site << '/' ;
			dir = ss.str() ;
			dir.insert(0, "mkdir -p ") ;
			system( dir.c_str() ) ;

			dir = ss.str() ;
			dir.insert( dir.length() , "calPSz.dat" ) ;

			ofstream	fsave( dir.c_str() , ios::app ) ;
			fsave << setprecision(16) << setw(19) << scientific ;
			for( unsigned int i=0 ; i<2*num_site ; i++ ) {
				fsave << opfOb.Sz[i] << "\t" ;
			}
			totSz   = 0 ;
			totStSz = 0 ;
			totStSzNedge = 0 ;
			for( unsigned int i=0 ; i<num_site ; i++ ) {
				totSz += opfOb.Sz[i] ;
				totSz += opfOb.Sz[i+num_site] ;
				totStSz += opfOb.Sz[2*i] ;
				totStSz -= opfOb.Sz[2*i+1] ;
				totStSzNedge += opfOb.Sz[2*i] ;
				totStSzNedge -= opfOb.Sz[2*i+1] ;
			}
			totStSzNedge -= opfOb.Sz[0] ;
			totStSzNedge += opfOb.Sz[2*num_site-1] ;
			fsave << endl ;
			fsave.close() ;
		}
		void calcPSzCorrWrite( char * oooutfdir ) 
		{

			stringstream ss ;
			string dir ;
			ss << oooutfdir << "/L" << num_site << '/' ;
			dir = ss.str() ;
			dir.insert(0, "mkdir -p ") ;
			system( dir.c_str() ) ;

			dir = ss.str() ;
			dir.insert( dir.length() , "calCorrPSz.dat" ) ;

			ofstream	fsave( dir.c_str() , ios::app ) ;
			double dumcorr = 0 ;
			for( size_t l=0 ; l<num_site ; l++ ) {
				size_t opInd =num_site-1-l ;
				dumcorr = opfOb.POpSzCorrGslSym( dum_gs , opInd ) ;
				fsave << setw(5) << opInd <<"\t" << 2*l+1 <<"\t" ;
				fsave << setprecision(16) << setw(19) << dumcorr << endl ;
			}
			fsave.close() ;
		}
		void calcPLocalSSWrite( char * oooutfdir ) 
		{
			opfOb.cenS = ssiteSop ; 
			opfOb.iterPLocalSSGsl( dum_gs , Jz ) ;


			stringstream ss ;
			string dir ;
			ss << oooutfdir << "/L" << num_site << '/' ;
			dir = ss.str() ;
			dir.insert(0, "mkdir -p ") ;
			system( dir.c_str() ) ;

			dir = ss.str() ;
			dir.insert( dir.length() , "calPLocalSS.dat" ) ;

			ofstream	fsave( dir.c_str() , ios::app ) ;
			for( size_t l=0 ; l<2*num_site-1 ; l++ ) {
				fsave << setprecision(16) << setw(19) << opfOb.PLocalSS[l] << endl ;
			}
			fsave.close() ;
		}
		size_t L()
	       	{
			return num_site ;
		}
} ;

class supSubBlockChi  : public sup_sub_block 
{
	public :
		double J1n ;
		void setInitSopChi( sopSpinHalfCoup spSopBlock , sopSpinOne spSopSsite , double JJend, double JJz , double JJ2 ) 
		{
			if ( num_site != 2 ) 
				printf("ERROR :: In 'setInitSopChi(..)', initial condition of L=2 is not matched.\n" ) ;
			Jz = JJz ;
			J2 = JJ2 * JJend ;
			sysBlockSop.allocCpy( spSopBlock ) ;
			ssiteSop.allocCpy   ( spSopSsite ) ;
			sparseMat dumB ,
				  dumC ;
			sysHamil.prodSop( sysBlockSop.Sz , ssiteSop.Sz , Jz * JJend ) ;
			dumB.prodSop( sysBlockSop.Sp , ssiteSop.Sm , 0.5*JJend) ;
			dumC.prodSop( sysBlockSop.Sm , ssiteSop.Sp , 0.5*JJend) ;

			sysHamil.addSop3( dumB ) ;
			sysHamil.addSop3( dumC ) ;
			dumB.sMatFree() ;
			dumC.sMatFree() ;
			tot_eval_gs_prev = 0 ; 

			opfOb.setDimL( num_site , num_block_state , num_ssite_state ) ;
			opfOb.initPOpSz ( ssiteSop.Sz , sysBlockSop.Sz , sysBlockSop2 ) ; // NO SS ;
			opfOb.initPOpChi( ssiteSop	 , sysBlockSop	  ) ;
		} //setInitSopChi()
		void opfSopLanczosDiagSymSupblock2( size_t num_tr , char * ooutfdir )
		{
			opfTotHamil.setAll( num_block_state , num_ssite_state , Jz , J2 , sysHamil , ssiteSop , sysBlockSop ) ;

			size_t dim_mat = num_sys_state * num_sys_state ;

			dum_gs     = gsl_vector_alloc ( dim_mat ) ;

			opfSopIterModLanczosRNlpPrev( opfTotHamil, dum_gs , tot_eval_gs , dim_mat , lanczosIter , deltaE , ooutfdir, num_site , tot_eval_gs_prev , nDecDigit ) ;

			gsl_matrix_view dum_mps = gsl_matrix_view_vector( dum_gs, num_sys_state, num_sys_state ) ;
			red_dm = gsl_matrix_calloc( num_sys_state, num_sys_state ) ;
			gsl_blas_dgemm( CblasTrans , CblasNoTrans ,
					1. , &dum_mps.matrix , &dum_mps.matrix ,
					0. , red_dm ) ;

			// Diagonalize the reduced density matrix(RDM) and sort by descending order of weight.
			gsl_eigen_symmv_workspace * w ;
			w = gsl_eigen_symmv_alloc( num_sys_state ) ;
			red_dm_eval = gsl_vector_alloc( num_sys_state ) ;
			red_dm_evec = gsl_matrix_alloc( num_sys_state , num_sys_state ) ;
			gsl_eigen_symmv( red_dm, red_dm_eval, red_dm_evec, w ) ;
			gsl_eigen_symmv_sort( red_dm_eval , red_dm_evec, GSL_EIGEN_SORT_VAL_DESC ) ;
			gsl_eigen_symmv_free( w ) ;

			// Set number of states going to be truncated.
			if( num_sys_state < num_tr ) 
				num_trunc_state = num_sys_state ;
			else
				num_trunc_state = num_tr ;

			// Before activating calc() options, must CHECK the update variables.

			cout <<"Lanczos::completed\n" << flush ;

		} //opfSopLanczosDiagSymSupblock2()
		void updateSop2( supSubBlockChi prev , double JJz , double JJ2 ) 
		{
			Jz = JJz ;
			J2 = JJ2 ;
			//gsl_vector_free( prev.dum_gs ) ;

			// Set Indicators.
			num_site = 1 + prev.num_site ;
			num_block_state = prev.num_trunc_state ;
			num_ssite_state = prev.num_ssite_state ;
			num_sys_state = num_block_state * num_ssite_state ;

			cout <<"update::L_sys to "<< num_site <<';' << flush ;
			sysBlockSop.Sz = prev.projectSop2aNz( prev.ssiteSop.Sz ) ;	//ssite to block
			sysBlockSop.Sp = prev.projectSop2aNz( prev.ssiteSop.Sp ) ;	//ssite to block
			sysBlockSop.Sm = prev.projectSop2aNz( prev.ssiteSop.Sm ) ;	//ssite to block
			sysBlockSop2.Sz = prev.projectSopSMatNz( prev.sysBlockSop.Sz ) ;	//prev.block to block
			sysBlockSop2.Sp = prev.projectSopSMatNz( prev.sysBlockSop.Sp ) ;	//prev.block to block
			sysBlockSop2.Sm = prev.projectSopSMatNz( prev.sysBlockSop.Sm ) ;	//prev.block to block
			sysBlockHamil   = prev.projectSopHamil2Nz( prev.sysHamil ) ;		//prev.sys to block
			ssiteSop = prev.ssiteSop ; 
			prev.sysBlockSop.sopCFree() ;

			tot_eval_gs_prev = prev.tot_eval_gs ; 

			opfOb.setDimL( num_site, num_block_state, num_ssite_state ) ;
			opfOb.updatePOpSz( ssiteSop.Sz , sysBlockSop.Sz , sysBlockSop2.Sz ) ;
			for( int i=0 ; i<(int)num_site-3 ; i++ )
			{
				opfOb.POpSz[i] = prev.projectSopSMatNz( prev.opfOb.POpSz[i] ) ;
			}
			prev.sysBlockSop2.sopCFree() ;

			// need only if calcPLocalSS
			//for( int i=0 ; i<(int)num_site-3 ; i++ )
			//{
				//opfOb.POpSS[i] = prev.projectSopSMatNz( prev.opfOb.POpSS[i] ) ;
			//}
			//opfOb.POpSS[num_site-3] = prev.projectSopHamil2Nz( prev.opfOb.POpSS[num_site-3] ) ;

			opfOb.updatePOpChi( ssiteSop , sysBlockSop ) ;
			for( int i=0 ; i<(int)num_site-3 ; i++ )
			{
				opfOb.POpChi[i] = prev.projectSopSMatNz( prev.opfOb.POpChi[i] ) ;
			}
			opfOb.POpChi[num_site-3] = prev.projectSopHamil2Nz( prev.opfOb.POpChi[num_site-3] ) ;

			prev.sysHamil.sMatFree() ;

			sysHamil.prodSopRId( sysBlockHamil , num_ssite_state ) ;
			sysBlockHamil.sMatFree() ;

			// For J1 (need whenever)
			sparseMat dumA ,
				  dumC ,
				  dumD ;
			dumA.prodSop( sysBlockSop.Sz , ssiteSop.Sz , Jz  ) ; sysHamil.addSop3( dumA ) ; dumA.sMatFree() ;
			dumC.prodSop( sysBlockSop.Sp , ssiteSop.Sm , 0.5 ) ; sysHamil.addSop3( dumC ) ; dumC.sMatFree() ;
			dumD.prodSop( sysBlockSop.Sm , ssiteSop.Sp , 0.5 ) ; sysHamil.addSop3( dumD ) ; dumD.sMatFree() ;

			// Options for PLocalSS (inserting into the paragraph above)
			//opfOb.POpSS[num_site-2].sMatAllocCpy( dumA );
			//opfOb.POpSS[num_site-2].addSop3( dumA ) ;
			//opfOb.POpSS[num_site-2].addSop3( dumC ) ;
			//opfOb.POpSS[num_site-2].addSop3( dumD ) ;
			//sysHamil.addSop3( opfOb.POpSS[num_site-2] ) ;
					
			// If PLocalSS considered Use freePOpSS()  etc.
			prev.opfOb.freePOpSz () ;
			prev.opfOb.freePOpChi() ;

			// For J2
			dumA.prodSop( sysBlockSop2.Sz , ssiteSop.Sz , J2*Jz  ) ; sysHamil.addSop3( dumA ) ; dumA.sMatFree() ;
			dumC.prodSop( sysBlockSop2.Sp , ssiteSop.Sm , J2*0.5 ) ; sysHamil.addSop3( dumC ) ; dumC.sMatFree() ;
			dumD.prodSop( sysBlockSop2.Sm , ssiteSop.Sp , J2*0.5 ) ; sysHamil.addSop3( dumD ) ; dumD.sMatFree() ;

			gsl_matrix_free( prev.red_dm_evec ) ;
		}
		void calcCPChiCorrWrite( char * oooutfdir ) 
		{

			stringstream ss ;
			string dir ;
			ss << oooutfdir << "/L" << num_site << '/' ;
			dir = ss.str() ;
			dir.insert(0, "mkdir -p ") ;
			system( dir.c_str() ) ;

			dir = ss.str() ;
			dir.insert( dir.length() , "calCorrPChi.dat" ) ;

			ofstream	fsave( dir.c_str() , ios::app ) ;
			double dumcorr = 0 ;
			for( int l=0 ; l<(int)num_site-1 ; l++ ) {
				int opInd =num_site-2-l ;
				dumcorr = opfOb.POpChiCorrGslSym( dum_gs , opInd , l ) ;
				fsave << setw(5) << opInd <<"\t" << 2*l+1 <<"\t" ;
				fsave << setprecision(16) << setw(19) << dumcorr << endl ;
				if( opInd > 0 ) 
				{
					dumcorr = opfOb.POpChiCorrGslAsym( dum_gs , opInd , l ) ;
					fsave << setw(5) << opInd <<"\t" << 2*l+2 <<"\t" ;
					fsave << setprecision(16) << setw(19) << dumcorr << endl ;
				}
			}
			fsave.close() ;
		}
		void calcCPSzWrite( char * oooutfdir ) 
		{
			opfOb.POpSzGsl( dum_gs ) ;

			stringstream ss ;
			string dir ;
			ss << oooutfdir << "/L" << num_site << '/' ;
			dir = ss.str() ;
			dir.insert(0, "mkdir -p ") ;
			system( dir.c_str() ) ;

			dir = ss.str() ;
			dir.insert( dir.length() , "calPSz.dat" ) ;

			ofstream	fsave( dir.c_str() , ios::app ) ;
			fsave << setprecision(16) << setw(19) << scientific ;
			for( unsigned int i=0 ; i<2*num_site ; i++ ) {
				fsave << opfOb.Sz[i] << "\t" ;
			}
			totSz   = 0 ;
			totStSz = 0 ;
			totStSzNedge = 0 ;
			for( unsigned int i=0 ; i<num_site ; i++ ) {
				totSz += opfOb.Sz[i] ;
				totSz += opfOb.Sz[i+num_site] ;
				totStSz += opfOb.Sz[2*i] ;
				totStSz -= opfOb.Sz[2*i+1] ;
				totStSzNedge += opfOb.Sz[2*i] ;
				totStSzNedge -= opfOb.Sz[2*i+1] ;
			}
			totStSzNedge -= opfOb.Sz[0] ;
			totStSzNedge += opfOb.Sz[2*num_site-1] ;
			fsave << endl ;
			fsave.close() ;
		}
		void calcCPSzCorrWrite( char * oooutfdir ) 
		{

			stringstream ss ;
			string dir ;
			ss << oooutfdir << "/L" << num_site << '/' ;
			dir = ss.str() ;
			dir.insert(0, "mkdir -p ") ;
			system( dir.c_str() ) ;

			dir = ss.str() ;
			dir.insert( dir.length() , "calCorrPSz.dat" ) ;

			ofstream	fsave( dir.c_str() , ios::app ) ;
			double dumcorr = 0 ;
			for( size_t l=0 ; l<num_site ; l++ ) {
				size_t opInd =num_site-1-l ;
				dumcorr = opfOb.POpSzCorrGslSym( dum_gs , opInd ) ;
				fsave << setw(5) << opInd <<"\t" << 2*l+1 <<"\t" ;
				fsave << setprecision(16) << setw(19) << dumcorr << endl ;
			}
			fsave.close() ;
		}
		void calcCPLocalSSWrite( char * oooutfdir ) 
		{
			opfOb.cenS = ssiteSop ; 
			opfOb.iterPLocalSSGsl( dum_gs , Jz ) ;

			stringstream ss ;
			string dir ;
			ss << oooutfdir << "/L" << num_site << '/' ;
			dir = ss.str() ;
			dir.insert(0, "mkdir -p ") ;
			system( dir.c_str() ) ;

			dir = ss.str() ;
			dir.insert( dir.length() , "calPLocalSS.dat" ) ;

			ofstream	fsave( dir.c_str() , ios::app ) ;
			for( size_t l=0 ; l<2*num_site-1 ; l++ ) {
				fsave << setprecision(16) << setw(19) << opfOb.PLocalSS[l] << endl ;
			}
			fsave.close() ;
		}
		void setInitSopChiNeg( sopSpinHalfCoup spSopBlock , sopSpinOne spSopSsite , double JJend, double JJz , double JJ2 , double JJ1 ) 
		{
			if ( num_site != 2 ) 
				printf("ERROR :: In 'setInitSopChi(..)', initial condition of L=2 is not matched.\n" ) ;
			Jz = JJz ;
			J2 = JJ2 * JJend ;
			J1n= -JJ1 ;
			sysBlockSop.allocCpy( spSopBlock ) ;
			ssiteSop.allocCpy   ( spSopSsite ) ;
			sparseMat dumB ,
				  dumC ;
			sysHamil.prodSop( sysBlockSop.Sz , ssiteSop.Sz , J1n * Jz * JJend ) ;
			dumB.prodSop( sysBlockSop.Sp , ssiteSop.Sm , J1n * 0.5*JJend) ;
			dumC.prodSop( sysBlockSop.Sm , ssiteSop.Sp , J1n * 0.5*JJend) ;

			sysHamil.addSop3( dumB ) ;
			sysHamil.addSop3( dumC ) ;
			dumB.sMatFree() ;
			dumC.sMatFree() ;
			tot_eval_gs_prev = 0 ; 

			opfOb.setDimL( num_site , num_block_state , num_ssite_state ) ;
			opfOb.initPOpSz ( ssiteSop.Sz , sysBlockSop.Sz , sysBlockSop2 ) ; // NO SS ;
			opfOb.initPOpChi( ssiteSop	 , sysBlockSop	  ) ;
		} //setInitSopChi()
		void setInitSopChiNegNoe( sopSpinOne spSopSsite , double JJend, double JJz , double JJ2 , double JJ1 ) 
		{
			if ( num_site != 2 ) 
				printf("ERROR :: In 'setInitSopChi(..)', initial condition of L=2 is not matched.\n" ) ;
			Jz = JJz ;
			J2 = JJ2 * JJend ;
			J1n= -JJ1 ;
			sysBlockSop.allocCpy( spSopSsite ) ;
			ssiteSop.allocCpy   ( spSopSsite ) ;
			sparseMat dumB ,
				  dumC ;
			sysHamil.prodSop( sysBlockSop.Sz , ssiteSop.Sz , J1n * Jz * JJend ) ;
			dumB.prodSop( sysBlockSop.Sp , ssiteSop.Sm , J1n * 0.5*JJend) ;
			dumC.prodSop( sysBlockSop.Sm , ssiteSop.Sp , J1n * 0.5*JJend) ;

			sysHamil.addSop3( dumB ) ;
			sysHamil.addSop3( dumC ) ;
			dumB.sMatFree() ;
			dumC.sMatFree() ;
			tot_eval_gs_prev = 0 ; 

			opfOb.setDimL( num_site , num_block_state , num_ssite_state ) ;
			opfOb.initPOpSz ( ssiteSop.Sz , sysBlockSop.Sz , sysBlockSop2 ) ; // NO SS ;
			opfOb.initPOpChi( ssiteSop	 , sysBlockSop	  ) ;
		} //setInitSopChi()
		void setInitSopChiNegBe( sopSpinOne spSopSsite , double JJend, double JJz , double JJ2 , double JJ1 ) 
		{
			if ( num_site != 2 ) 
				printf("ERROR :: In 'setInitSopChi(..)', initial condition of L=2 is not matched.\n" ) ;
			Jz = JJz ;
			J2 = JJ2 ;
			J1n= -JJ1 ;
			sysBlockSop.allocCpy( spSopSsite ) ;
			ssiteSop.allocCpy   ( spSopSsite ) ;
			sparseMat dumB ,
				  dumC ;
			sysHamil.prodSop( sysBlockSop.Sz , ssiteSop.Sz , J2 * Jz * JJend ) ;	// antiferromagnetically interact with both sites.
			dumB.prodSop( sysBlockSop.Sp , ssiteSop.Sm , J2 * 0.5*JJend) ;	// antiferromagnetically interact with both sites.
			dumC.prodSop( sysBlockSop.Sm , ssiteSop.Sp , J2 * 0.5*JJend) ;	// antiferromagnetically interact with both sites.

			sysHamil.addSop3( dumB ) ;
			sysHamil.addSop3( dumC ) ;
			dumB.sMatFree() ;
			dumC.sMatFree() ;
			tot_eval_gs_prev = 0 ; 

			opfOb.setDimL( num_site , num_block_state , num_ssite_state ) ;
			opfOb.initPOpSz ( ssiteSop.Sz , sysBlockSop.Sz , sysBlockSop2 ) ; // NO SS ;
			opfOb.initPOpChi( ssiteSop	 , sysBlockSop	  ) ;
		} //setInitSopChi()
		void updateSop2Neg( supSubBlockChi prev , double JJz , double JJ2 , double JJ1 ) 
		{
			Jz = JJz ;
			J2 = JJ2 ;
			J1n = -JJ1 ;
			//gsl_vector_free( prev.dum_gs ) ;

			// Set Indicators.
			num_site = 1 + prev.num_site ;
			num_block_state = prev.num_trunc_state ;
			num_ssite_state = prev.num_ssite_state ;
			num_sys_state = num_block_state * num_ssite_state ;

			cout <<"update::L_sys to "<< num_site <<';' << flush ;
			sysBlockSop.Sz = prev.projectSop2aNz( prev.ssiteSop.Sz ) ;	//ssite to block
			sysBlockSop.Sp = prev.projectSop2aNz( prev.ssiteSop.Sp ) ;	//ssite to block
			sysBlockSop.Sm = prev.projectSop2aNz( prev.ssiteSop.Sm ) ;	//ssite to block
			sysBlockSop2.Sz = prev.projectSopSMatNz( prev.sysBlockSop.Sz ) ;	//prev.block to block
			sysBlockSop2.Sp = prev.projectSopSMatNz( prev.sysBlockSop.Sp ) ;	//prev.block to block
			sysBlockSop2.Sm = prev.projectSopSMatNz( prev.sysBlockSop.Sm ) ;	//prev.block to block
			sysBlockHamil   = prev.projectSopHamil2Nz( prev.sysHamil ) ;		//prev.sys to block
			ssiteSop = prev.ssiteSop ; 
			prev.sysBlockSop.sopCFree() ;

			tot_eval_gs_prev = prev.tot_eval_gs ; 

			opfOb.setDimL( num_site, num_block_state, num_ssite_state ) ;
			opfOb.updatePOpSz( ssiteSop.Sz , sysBlockSop.Sz , sysBlockSop2.Sz ) ;
			for( int i=0 ; i<(int)num_site-3 ; i++ )
			{
				opfOb.POpSz[i] = prev.projectSopSMatNz( prev.opfOb.POpSz[i] ) ;
			}
			prev.sysBlockSop2.sopCFree() ;

			// need only if calcPLocalSS
			//for( int i=0 ; i<(int)num_site-3 ; i++ )
			//{
				//opfOb.POpSS[i] = prev.projectSopSMatNz( prev.opfOb.POpSS[i] ) ;
			//}
			//opfOb.POpSS[num_site-3] = prev.projectSopHamil2Nz( prev.opfOb.POpSS[num_site-3] ) ;

			opfOb.updatePOpChi( ssiteSop , sysBlockSop ) ;
			for( int i=0 ; i<(int)num_site-3 ; i++ )
			{
				opfOb.POpChi[i] = prev.projectSopSMatNz( prev.opfOb.POpChi[i] ) ;
			}
			opfOb.POpChi[num_site-3] = prev.projectSopHamil2Nz( prev.opfOb.POpChi[num_site-3] ) ;

			prev.sysHamil.sMatFree() ;

			sysHamil.prodSopRId( sysBlockHamil , num_ssite_state ) ;
			sysBlockHamil.sMatFree() ;

			// For J1 (need whenever)
			sparseMat dumA ,
				  dumC ,
				  dumD ;
			dumA.prodSop( sysBlockSop.Sz , ssiteSop.Sz , J1n * Jz  ) ; sysHamil.addSop3( dumA ) ; dumA.sMatFree() ;
			dumC.prodSop( sysBlockSop.Sp , ssiteSop.Sm , J1n * 0.5 ) ; sysHamil.addSop3( dumC ) ; dumC.sMatFree() ;
			dumD.prodSop( sysBlockSop.Sm , ssiteSop.Sp , J1n * 0.5 ) ; sysHamil.addSop3( dumD ) ; dumD.sMatFree() ;

			// Options for PLocalSS (inserting into the paragraph above)
			//opfOb.POpSS[num_site-2].sMatAllocCpy( dumA );
			//opfOb.POpSS[num_site-2].addSop3( dumA ) ;
			//opfOb.POpSS[num_site-2].addSop3( dumC ) ;
			//opfOb.POpSS[num_site-2].addSop3( dumD ) ;
			//sysHamil.addSop3( opfOb.POpSS[num_site-2] ) ;
					
			// If PLocalSS considered Use freePOpSS()  etc.
			prev.opfOb.freePOpSz () ;
			prev.opfOb.freePOpChi() ;

			// For J2
			dumA.prodSop( sysBlockSop2.Sz , ssiteSop.Sz , J2*Jz  ) ; sysHamil.addSop3( dumA ) ; dumA.sMatFree() ;
			dumC.prodSop( sysBlockSop2.Sp , ssiteSop.Sm , J2*0.5 ) ; sysHamil.addSop3( dumC ) ; dumC.sMatFree() ;
			dumD.prodSop( sysBlockSop2.Sm , ssiteSop.Sp , J2*0.5 ) ; sysHamil.addSop3( dumD ) ; dumD.sMatFree() ;

			gsl_matrix_free( prev.red_dm_evec ) ;
		}
		void opfSopLanczosDiagSymSupblock2Neg( size_t num_tr , char * ooutfdir )
		{
			opfTotHamil.setAll( num_block_state , num_ssite_state , Jz , J2 , sysHamil , ssiteSop , sysBlockSop , J1n ) ;

			size_t dim_mat = num_sys_state * num_sys_state ;

			dum_gs     = gsl_vector_alloc ( dim_mat ) ;
			for( unsigned int ii=0 ; ii<dim_mat ; ii++ ) 
				gsl_vector_set( dum_gs , ii , rand()/double(RAND_MAX) - 0.5 ) ;
			gsl_vector_scale( dum_gs, 1./gsl_blas_dnrm2(dum_gs) ) ;

			opfSopIterModLanczosRNlpPrev( opfTotHamil, dum_gs , tot_eval_gs , dim_mat , lanczosIter , deltaE , ooutfdir, num_site , tot_eval_gs_prev , nDecDigit ) ;

			gsl_matrix_view dum_mps = gsl_matrix_view_vector( dum_gs, num_sys_state, num_sys_state ) ;
			red_dm = gsl_matrix_calloc( num_sys_state, num_sys_state ) ;
			gsl_blas_dgemm( CblasTrans , CblasNoTrans ,
					1. , &dum_mps.matrix , &dum_mps.matrix ,
					0. , red_dm ) ;

			// Diagonalize the reduced density matrix(RDM) and sort by descending order of weight.
			gsl_eigen_symmv_workspace * w ;
			w = gsl_eigen_symmv_alloc( num_sys_state ) ;
			red_dm_eval = gsl_vector_alloc( num_sys_state ) ;
			red_dm_evec = gsl_matrix_alloc( num_sys_state , num_sys_state ) ;
			gsl_eigen_symmv( red_dm, red_dm_eval, red_dm_evec, w ) ;
			gsl_eigen_symmv_sort( red_dm_eval , red_dm_evec, GSL_EIGEN_SORT_VAL_DESC ) ;
			gsl_eigen_symmv_free( w ) ;

			// Set number of states going to be truncated.
			if( num_sys_state < num_tr ) 
				num_trunc_state = num_sys_state ;
			else
				num_trunc_state = num_tr ;

			// Before activating calc() options, must CHECK the update variables.

			cout <<"Lanczos::completed\n" << flush ;

		} //opfSopLanczosDiagSymSupblock2()
		void opfSopLanczosDiagSymSupblock2Neg( size_t num_tr , char * ooutfdir , supSubBlockChi prev )
		{
			opfTotHamil.setAll( num_block_state , num_ssite_state , Jz , J2 , sysHamil , ssiteSop , sysBlockSop , J1n ) ;

			size_t dim_mat = num_sys_state * num_sys_state ;
			size_t num_blockc_state = num_sys_state * num_block_state ;
			size_t p_num_blockc_state = prev.num_sys_state * prev.num_block_state ;

			dum_gs     = gsl_vector_alloc ( dim_mat ) ; 


			gsl_matrix * dm_ssite = gsl_matrix_alloc(prev.num_ssite_state,prev.num_ssite_state) ;
			for( unsigned int bi=0 ; bi<prev.num_ssite_state ; bi++ ) 
			{
				for( unsigned int bj=0 ; bj<prev.num_ssite_state ; bj++ ) 
				{
					double dum_sij=0 ;
					for( unsigned int ai=0 ; ai<prev.num_block_state ; ai++ ) 
					{
						for( unsigned int ci=0 ; ci<prev.num_block_state ; ci++ ) 
						{
							for( unsigned int di=0 ; di<prev.num_ssite_state ; di++ ) 
							{
								size_t si   =  ai + prev.num_block_state*bi + prev.num_sys_state*ci + p_num_blockc_state*di ;
								size_t sj   =  ai + prev.num_block_state*bj + prev.num_sys_state*ci + p_num_blockc_state*di ;
								dum_sij += gsl_vector_get( prev.dum_gs, si ) * gsl_vector_get( prev.dum_gs, sj ) ;
							}
						}
					}
					gsl_matrix_set( dm_ssite, bi,bj , dum_sij ) ;
				}
			}
			gsl_vector * dm_ssite_eval = gsl_vector_alloc( prev.num_ssite_state ) ;
			gsl_matrix * dm_ssite_evec = gsl_matrix_alloc( prev.num_ssite_state , prev.num_ssite_state ) ;
			gsl_eigen_symmv_workspace * w2 ;
			w2 = gsl_eigen_symmv_alloc( prev.num_ssite_state ) ;
			gsl_eigen_symmv( dm_ssite, dm_ssite_eval, dm_ssite_evec, w2 ) ;
			gsl_eigen_symmv_sort( dm_ssite_eval , dm_ssite_evec, GSL_EIGEN_SORT_VAL_DESC ) ;
			gsl_eigen_symmv_free( w2 ) ;

			double *bdum = new double[3] ,
			       *adum = new double[num_block_state] ;
			srand(1) ;
			for( unsigned int bi=0 ; bi<num_ssite_state ; bi++ )
				bdum[bi] = 0 ;
			for( unsigned int bi=0 ; bi<num_ssite_state ; bi++ )
			{
				int rsign = rand()%2*2-1 ;
				double sqrtw = gsl_vector_get(dm_ssite_eval,bi) ;
				if( sqrtw < 0 )
					sqrtw = 0 ;
				else
					sqrtw = sqrt(sqrtw) ;
				bdum[0] += double(rsign) * sqrtw * gsl_matrix_get( dm_ssite_evec ,0, bi) ;
				bdum[1] += double(rsign) * sqrtw * gsl_matrix_get( dm_ssite_evec ,1, bi) ;
				bdum[2] += double(rsign) * sqrtw * gsl_matrix_get( dm_ssite_evec ,2, bi) ;
			}
			for( unsigned int ai=0 ; ai<num_block_state ; ai++ ) 
			{
				int rsign = rand()%2*2-1 ;
				double sqrtw = gsl_vector_get(prev.red_dm_eval , ai) ;
				if( sqrtw < 0 )
					sqrtw = 0 ;
				else
					sqrtw = sqrt(sqrtw) ;
				adum[ai] =  double(rsign) * sqrtw ;
			}

			for( unsigned int ai=0 ; ai<num_block_state ; ai++ ) {
				for( unsigned int ci=0 ; ci<num_block_state ; ci++ ) {
					for( unsigned int di=0 ; di<num_ssite_state ; di++ ) {
						for( unsigned int bi=0 ; bi<num_ssite_state ; bi++ ) {
							size_t ii   =  ai + num_block_state*bi + num_sys_state*ci + num_blockc_state*di ;
							gsl_vector_set( dum_gs,ii , adum[ai]*adum[ci]*bdum[bi]*bdum[di] ) ;
						}
					}
				}
			}
			//for( size_t ii=0 ; ii<dum_gs->size ; ii++ )
				//cout <<'['<<ii<<']'<< gsl_vector_get(dum_gs,ii) << endl ;
			gsl_vector_free( dm_ssite_eval ) ;
			gsl_matrix_free( dm_ssite_evec ) ;
			gsl_matrix_free( dm_ssite ) ;
			gsl_vector_free( prev.dum_gs ) ;
			delete []adum ;
			delete []bdum ;
			gsl_vector_scale( dum_gs, 1./gsl_blas_dnrm2(dum_gs) ) ;
			gsl_vector_free( prev.red_dm_eval ) ;

			opfSopIterModLanczosRNlpPrev( opfTotHamil, dum_gs , tot_eval_gs , dim_mat , lanczosIter , deltaE , ooutfdir, num_site , tot_eval_gs_prev , nDecDigit ) ;

			gsl_matrix_view dum_mps = gsl_matrix_view_vector( dum_gs, num_sys_state, num_sys_state ) ;
			red_dm = gsl_matrix_calloc( num_sys_state, num_sys_state ) ;
			gsl_blas_dgemm( CblasTrans , CblasNoTrans ,
					1. , &dum_mps.matrix , &dum_mps.matrix ,
					0. , red_dm ) ;

			// Diagonalize the reduced density matrix(RDM) and sort by descending order of weight.
			gsl_eigen_symmv_workspace * w ;
			w = gsl_eigen_symmv_alloc( num_sys_state ) ;
			red_dm_eval = gsl_vector_alloc( num_sys_state ) ;
			red_dm_evec = gsl_matrix_alloc( num_sys_state , num_sys_state ) ;
			gsl_eigen_symmv( red_dm, red_dm_eval, red_dm_evec, w ) ;
			gsl_eigen_symmv_sort( red_dm_eval , red_dm_evec, GSL_EIGEN_SORT_VAL_DESC ) ;
			gsl_eigen_symmv_free( w ) ;

			// Set number of states going to be truncated.
			if( num_sys_state < num_tr ) 
				num_trunc_state = num_sys_state ;
			else
				num_trunc_state = num_tr ;

			// Before activating calc() options, must CHECK the update variables.

			cout <<"Lanczos::completed\n" << flush ;

		} //opfSopLanczosDiagSymSupblock2()
		void setJ1n( double j1 )
		{
			J1n = -j1 ;
		}
} ;
class supSubBlockChiS  : public supSubBlockChi
{
	public :
		void setInitPop()
		{
			opfOb.initPOpSxy( ssiteSop , sysBlockSop ) ;
			opfOb.initPOpStr( ssiteSop , sysBlockSop ) ;
			opfOb.initPOpSstr( ssiteSop , sysBlockSop ) ;
		}
		void calcPSxyCorr( char * ooutfdir ) 
		{
			stringstream ss ; 
			string dir ;
			ss << ooutfdir << "/L" << num_site << '/' ;
			dir = ss.str() ;
			dir.insert(0, "mkdir -p ") ;
			system( dir.c_str() ) ;

			dir = ss.str() ;
			dir.insert( dir.length() , "calCorrPSxy.dat" ) ;

			ofstream        fsave( dir.c_str() , ios::app ) ;
			double dumcorr = 0 ;
			for( size_t l=0 ; l<num_site ; l++ ) {
				size_t opInd =num_site-1-l ;
				dumcorr = opfOb.POpSxyCorrGslSym( dum_gs , opInd ) ;
				fsave << setw(5) << opInd <<"\t" << 2*l+1 <<"\t" ;
				fsave << setprecision(16) << setw(19) << dumcorr << endl ;
			}    
			fsave.close() ;
		}
		void calcPSxyCorrE( char * ooutfdir ) 
		{
			stringstream ss ; 
			string dir ;
			ss << ooutfdir << "/L" << num_site << '/' ;
			dir = ss.str() ;
			dir.insert(0, "mkdir -p ") ;
			system( dir.c_str() ) ;

			dir = ss.str() ;
			dir.insert( dir.length() , "calCorrPSxy.dat" ) ;

			ofstream        fsave( dir.c_str() , ios::app ) ;
			double dumcorr = 0 ;
			for( size_t l=0 ; l<num_site ; l++ ) {
				size_t opInd =num_site-1-l ;
				dumcorr = opfOb.POpSxyCorrGslSym( dum_gs , opInd ) ;
				fsave << setw(5) << opInd <<"\t" << 2*l+1 <<"\t" ;
				fsave << setprecision(16) << setw(19) << dumcorr << endl ;
				if( opInd > 0 )
				{
				dumcorr = opfOb.POpSxyCorrGslSymE( dum_gs , opInd ) ;
				fsave << setw(5) << opInd <<"\t" << 2*l+2 <<"\t" ;
				fsave << setprecision(16) << setw(19) << dumcorr << endl ;
				}
			}    
			fsave.close() ;
		}
		void calcPStotCorr( char * ooutfdir ) 
		{
			stringstream ss ; 
			string dir ;
			ss << ooutfdir << "/L" << num_site << '/' ;
			dir = ss.str() ;
			dir.insert(0, "mkdir -p ") ;
			system( dir.c_str() ) ;

			dir = ss.str() ;
			dir.insert( dir.length() , "calCorrPStot.dat" ) ;

			ofstream        fsave( dir.c_str() , ios::app ) ;
			double dumcorr = 0 ;
			for( size_t l=0 ; l<num_site ; l++ ) {
				size_t opInd =num_site-1-l ;
				dumcorr = opfOb.POpStotCorrGslSym( dum_gs , opInd ) ;
				fsave << setw(5) << opInd <<"\t" << 2*l+1 <<"\t" ;
				fsave << setprecision(16) << setw(19) << dumcorr << endl ;
			}    
			fsave.close() ;
		}
		void calcPStotCorrE( char * ooutfdir ) 
		{
			stringstream ss ; 
			string dir ;
			ss << ooutfdir << "/L" << num_site << '/' ;
			dir = ss.str() ;
			dir.insert(0, "mkdir -p ") ;
			system( dir.c_str() ) ;

			dir = ss.str() ;
			dir.insert( dir.length() , "calCorrPStot.dat" ) ;

			ofstream        fsave( dir.c_str() , ios::app ) ;
			double dumcorr = 0 ;
			for( size_t l=0 ; l<num_site ; l++ ) {
				size_t opInd =num_site-1-l ;
				dumcorr = opfOb.POpStotCorrGslSym( dum_gs , opInd ) ;
				fsave << setw(5) << opInd <<"\t" << 2*l+1 <<"\t" ;
				fsave << setprecision(16) << setw(19) << dumcorr << endl ;
				if( opInd > 0 )
				{
					dumcorr = opfOb.POpStotCorrGslSymE( dum_gs , opInd ) ;
					fsave << setw(5) << opInd <<"\t" << 2*l+2 <<"\t" ;
					fsave << setprecision(16) << setw(19) << dumcorr << endl ;
				}
			}    
			fsave.close() ;
		}
		void calcPStot( char * ooutfdir ) 
		{
			stringstream ss ; 
			string dir ;
			ss << ooutfdir << "/L" << num_site << '/' ;
			dir = ss.str() ;
			dir.insert(0, "mkdir -p ") ;
			system( dir.c_str() ) ;

			dir = ss.str() ;
			dir.insert( dir.length() , "calPStot.dat" ) ;

			ofstream        fsave( dir.c_str() , ios::app ) ;
			stotcomp = (double *) malloc( sizeof(double)*3 ) ;
			opfOb.POpStot2( dum_gs , stotcomp ) ;
			double dum = stotcomp[0] + stotcomp[1] + stotcomp[2] ;
			fsave << num_site << "\t" << dum << "\t" << -0.5+0.5*sqrt(1+4*dum) << endl ;
			fsave.close() ;
		}
		void calcPSxzStrCorrE( char * ooutfdir ) 
		{
			stringstream ss ; 
			string dir ,
			       dirx ;
			ss << ooutfdir << "/L" << num_site << '/' ;
			dir = ss.str() ;
			dir.insert(0, "mkdir -p ") ;
			system( dir.c_str() ) ;

			dir = ss.str() ;
			dirx= ss.str() ;
			dir.insert( dir.length() , "calStrPSz_dE.dat" ) ;
			dirx.insert( dirx.length() , "calStrPSx_dE.dat" ) ;

			ofstream        fsave( dir.c_str() , ios::app ) ;
			ofstream        fsavex( dirx.c_str() , ios::app ) ;
			int num_str = 2*num_site - 3 ;
			for( int l=0 ; l<num_str ; l++ ) {
				double str[5] ;
				double strx[1] ;
				if( l%2 < 1 )
				{
					opfOb.POpSzStrCorrGslSymD( dum_gs , l , num_site, str ) ;
					opfOb.POpSxStrCorrGslSymD( dum_gs , l , num_site, strx) ;
				fsave << setw(5) << l <<"\t" ;
				fsave << setprecision(16) << setw(19) << str[0] << endl << flush ;
				fsavex<< setw(5) << l <<"\t" ;
				fsavex<< setprecision(16) << setw(19) << strx[0] << endl << flush ;
				}
				else
				{
					opfOb.POpSzStrCorrGslAsymD( dum_gs , l , num_site, str ) ;
					opfOb.POpSxStrCorrGslAsymD( dum_gs , l , num_site, strx) ;
				fsave << setw(5) << l <<"\t" ;
				fsave << setprecision(16) << setw(19) << str[0] << endl << flush ;
				fsavex<< setw(5) << l <<"\t" ;
				fsavex<< setprecision(16) << setw(19) << strx[0] << endl << flush ;
				}
			}    
			fsave.close() ;
			fsavex.close() ;
		}
		void calcPSxzStrCorr( char * ooutfdir ) 
		{
			stringstream ss ; 
			string dir ,
			       dirx ;
			ss << ooutfdir << "/L" << num_site << '/' ;
			dir = ss.str() ;
			dir.insert(0, "mkdir -p ") ;
			system( dir.c_str() ) ;

			dir = ss.str() ;
			dirx= ss.str() ;
			dir.insert( dir.length() , "calStrPSz_d.dat" ) ;
			dirx.insert( dirx.length() , "calStrPSx_d.dat" ) ;

			ofstream        fsave( dir.c_str() , ios::app ) ;
			ofstream        fsavex( dirx.c_str() , ios::app ) ;
			int num_str = 2*num_site - 3 ;
			for( int l=0 ; l<num_str ; l++ ) {
				double str[5] ;
				double strx[1] ;
				if( l%2 < 1 )
				{
					opfOb.POpSzStrCorrGslSymD( dum_gs , l , num_site, str ) ;
					opfOb.POpSxStrCorrGslSymD( dum_gs , l , num_site, strx) ;
				//else
					//opfOb.POpSzStrCorrGslAsym(dum_gs , l , num_site, str ) ;
				fsave << setw(5) << l <<"\t" ;
				fsave << setprecision(16) << setw(19) << str[0] << endl << flush ;
				fsavex<< setw(5) << l <<"\t" ;
				fsavex<< setprecision(16) << setw(19) << strx[0] << endl << flush ;
				}
			}    
			fsave.close() ;
			fsavex.close() ;
		}
		void calcPSzStrCorr( char * ooutfdir ) 
		{
			stringstream ss ; 
			string dir ,
			       dirx ;
			ss << ooutfdir << "/L" << num_site << '/' ;
			dir = ss.str() ;
			dir.insert(0, "mkdir -p ") ;
			system( dir.c_str() ) ;

			dir = ss.str() ;
			dir.insert( dir.length() , "calStrPSz_d.dat" ) ;

			ofstream        fsave( dir.c_str() , ios::app ) ;
			int num_str = 2*num_site - 3 ;
			for( int l=0 ; l<num_str ; l++ ) {
				double str[5] ;
				if( l%2 < 1 )
				{
					opfOb.POpSzStrCorrGslSymD( dum_gs , l , num_site, str ) ;
				//else
					//opfOb.POpSzStrCorrGslAsym(dum_gs , l , num_site, str ) ;
				fsave << setw(5) << l <<"\t" ;
				fsave << setprecision(16) << setw(19) << str[0] << endl << flush ;
				}
			}    
			fsave.close() ;
		}
		void calcPSxzSstrCorrE( char * ooutfdir ) 
		{
			stringstream ss ; 
			string dir ,
			       dirx ;
			ss << ooutfdir << "/L" << num_site << '/' ;
			dir = ss.str() ;
			dir.insert(0, "mkdir -p ") ;
			system( dir.c_str() ) ;

			dir = ss.str() ;
			dirx= ss.str() ;
			dir.insert( dir.length() , "calSstrPSz_dE.dat" ) ;
			dirx.insert( dirx.length() , "calSstrPSx_dE.dat" ) ;

			ofstream        fsave( dir.c_str() , ios::app ) ;
			ofstream        fsavex( dirx.c_str() , ios::app ) ;
			int num_str = 2*num_site - 1 ;
			for( int l=0 ; l<num_str ; l++ ) {
				double str[5] ;
				double strx[1] ;
				if( l%2 < 1 )
				{
					opfOb.POpSzSstrCorrGslSymD( dum_gs , l , num_site, str ) ;
					opfOb.POpSxSstrCorrGslSymD( dum_gs , l , num_site, strx) ;
				//else
					//opfOb.POpSzStrCorrGslAsym(dum_gs , l , num_site, str ) ;
				fsave << setw(5) << l <<"\t" ;
				fsave << setprecision(16) << setw(19) << str[0] << endl ;
				fsavex<< setw(5) << l <<"\t" ;
				fsavex<< setprecision(16) << setw(19) << strx[0] << endl ;
				}
				else
				{
					opfOb.POpSzSstrCorrGslAsymD( dum_gs , l , num_site, str ) ;
					opfOb.POpSxSstrCorrGslAsymD( dum_gs , l , num_site, strx) ;
				//else
					//opfOb.POpSzStrCorrGslAsym(dum_gs , l , num_site, str ) ;
				fsave << setw(5) << l <<"\t" ;
				fsave << setprecision(16) << setw(19) << str[0] << endl ;
				fsavex<< setw(5) << l <<"\t" ;
				fsavex<< setprecision(16) << setw(19) << strx[0] << endl ;
				}
			}    
			fsave.close() ;
			fsavex.close() ;
		}
		void calcPSxzSstrCorr( char * ooutfdir ) 
		{
			stringstream ss ; 
			string dir ,
			       dirx ;
			ss << ooutfdir << "/L" << num_site << '/' ;
			dir = ss.str() ;
			dir.insert(0, "mkdir -p ") ;
			system( dir.c_str() ) ;

			dir = ss.str() ;
			dirx= ss.str() ;
			dir.insert( dir.length() , "calSstrPSz_d.dat" ) ;
			dirx.insert( dirx.length() , "calSstrPSx_d.dat" ) ;

			ofstream        fsave( dir.c_str() , ios::app ) ;
			ofstream        fsavex( dirx.c_str() , ios::app ) ;
			int num_str = 2*num_site - 1 ;
			for( int l=0 ; l<num_str ; l++ ) {
				double str[5] ;
				double strx[1] ;
				if( l%2 < 1 )
				{
					opfOb.POpSzSstrCorrGslSymD( dum_gs , l , num_site, str ) ;
					opfOb.POpSxSstrCorrGslSymD( dum_gs , l , num_site, strx) ;
				//else
					//opfOb.POpSzStrCorrGslAsym(dum_gs , l , num_site, str ) ;
				fsave << setw(5) << l <<"\t" ;
				fsave << setprecision(16) << setw(19) << str[0] << endl ;
				fsavex<< setw(5) << l <<"\t" ;
				fsavex<< setprecision(16) << setw(19) << strx[0] << endl ;
				}
			}    
			fsave.close() ;
			fsavex.close() ;
		}
		void calcPSzSstrCorr( char * ooutfdir ) 
		{
			stringstream ss ; 
			string dir ;
			ss << ooutfdir << "/L" << num_site << '/' ;
			dir = ss.str() ;
			dir.insert(0, "mkdir -p ") ;
			system( dir.c_str() ) ;

			dir = ss.str() ;
			dir.insert( dir.length() , "calSstrPSz_d.dat" ) ;

			ofstream        fsave( dir.c_str() , ios::app ) ;
			int num_str = 2*num_site - 1 ;
			for( int l=0 ; l<num_str ; l++ ) {
				double str[5] ;
				if( l%2 < 1 )
				{
					opfOb.POpSzSstrCorrGslSymD( dum_gs , l , num_site, str ) ;
				//else
					//opfOb.POpSzStrCorrGslAsym(dum_gs , l , num_site, str ) ;
				fsave << setw(5) << l <<"\t" ;
				fsave << setprecision(16) << setw(19) << str[0] << endl ;
				}
			}    
			fsave.close() ;
		}
		void updateSop2NegS( supSubBlockChi prev , double JJz , double JJ2 , double JJ1 ) 
		{
			Jz = JJz ;
			J2 = JJ2 ;
			J1n = -JJ1 ;

			// Set Indicators.
			num_site = 1 + prev.num_site ;
			num_block_state = prev.num_trunc_state ;
			num_ssite_state = prev.num_ssite_state ;
			num_sys_state = num_block_state * num_ssite_state ;

			cout <<"update::L_sys to "<< num_site <<';' << flush ;
			sysBlockSop.Sz = prev.projectSop2aNz( prev.ssiteSop.Sz ) ;	//ssite to block
			sysBlockSop.Sp = prev.projectSop2aNz( prev.ssiteSop.Sp ) ;	//ssite to block
			sysBlockSop.Sm = prev.projectSop2aNz( prev.ssiteSop.Sm ) ;	//ssite to block
			sysBlockSop2.Sz = prev.projectSopSMatNz( prev.sysBlockSop.Sz ) ;	//prev.block to block
			sysBlockSop2.Sp = prev.projectSopSMatNz( prev.sysBlockSop.Sp ) ;	//prev.block to block
			sysBlockSop2.Sm = prev.projectSopSMatNz( prev.sysBlockSop.Sm ) ;	//prev.block to block
			sysBlockHamil   = prev.projectSopHamil2Nz( prev.sysHamil ) ;		//prev.sys to block
			ssiteSop = prev.ssiteSop ; 
			prev.sysBlockSop.sopCFree() ;

			tot_eval_gs_prev = prev.tot_eval_gs ; 

			opfOb.setDimL( num_site, num_block_state, num_ssite_state ) ;
			opfOb.updatePOpSz ( ssiteSop.Sz , sysBlockSop.Sz , sysBlockSop2.Sz ) ;
			opfOb.updatePOpSxy( ssiteSop    , sysBlockSop    , sysBlockSop2    ) ;
			for( int i=0 ; i<(int)num_site-3 ; i++ )
			{
				opfOb.POpSz[i] = prev.projectSopSMatNz( prev.opfOb.POpSz[i] ) ;
				opfOb.POpSp[i] = prev.projectSopSMatNz( prev.opfOb.POpSp[i] ) ;
				opfOb.POpSm[i] = prev.projectSopSMatNz( prev.opfOb.POpSm[i] ) ;
			}
			prev.sysBlockSop2.sopCFree() ;

			// need only if calcPLocalSS
			//for( int i=0 ; i<(int)num_site-3 ; i++ )
			//{
				//opfOb.POpSS[i] = prev.projectSopSMatNz( prev.opfOb.POpSS[i] ) ;
			//}
			//opfOb.POpSS[num_site-3] = prev.projectSopHamil2Nz( prev.opfOb.POpSS[num_site-3] ) ;

			//calc op for the double string order //// PD[] = PDotGsl_type
			opfOb.updatePOpStr( ssiteSop , sysBlockSop ) ;	//L-2 PD4,PD5
			opfOb.POpStrSz[num_site-3] = prev.projectSopHamil2Nz( prev.opfOb.POpStrSz[num_site-3] ) ;	//L-3 PD0,PD2
			opfOb.POpStrSx[num_site-3] = prev.projectSopHamil2Nz( prev.opfOb.POpStrSx[num_site-3] ) ;	//L-3 PD0,PD2
			sparseMat dum ,
				  dum2 ;
			sparseMat dumx ,
				  dumx2 ;
			dum.setM(3,3) ;	//PD1,PD3
			dum.sDat[0].setDat( 0,0, -1 ) ;
			dum.sDat[1].setDat( 1,1,  1 ) ;
			dum.sDat[2].setDat( 2,2, -1 ) ;
			dumx.setM(3,3) ;	//PD1,PD3
			dumx.sDat[0].setDat( 0,2, -1 ) ;
			dumx.sDat[1].setDat( 1,1, -1 ) ;
			dumx.sDat[2].setDat( 2,0, -1 ) ;
			for( int i=0 ; i<(int)num_site-3 ; i++ )
			{
				dum2.prodSop( prev.opfOb.POpStrSz[i] , dum ) ;
				opfOb.POpStrSz[i] = prev.projectSopHamil2Nz( dum2 ) ;	// less than L-4 PD0,2
				dum2.sMatFree() ;

				dum2.prodSop( prev.opfOb.POpStrSx[i] , dumx ) ;
				opfOb.POpStrSx[i] = prev.projectSopHamil2Nz( dum2 ) ;	// less than L-4 PD0,2
				dum2.sMatFree() ;
			}
			////////

			//calc op for the single string order //// PD[] = PDotGsl_type
			opfOb.initPOpSstr( ssiteSop , sysBlockSop ) ;	//L-1,L-2 
			for( int i=0 ; i<(int)num_site-2 ; i++ )
			{
				dum2.prodSop( prev.opfOb.POpSstrSz[i] , dum ) ;
				opfOb.POpSstrSz[i] = prev.projectSopHamil2Nz( dum2 ) ;	// less than L-3 PD0,2
				dum2.sMatFree() ;

				dum2.prodSop( prev.opfOb.POpSstrSx[i] , dumx ) ;
				opfOb.POpSstrSx[i] = prev.projectSopHamil2Nz( dum2 ) ;	// less than L-3 PD0,2
				dum2.sMatFree() ;
			}
			dum.sMatFree() ;
			dumx.sMatFree() ;
			////////

			opfOb.updatePOpChi( ssiteSop , sysBlockSop ) ;
			for( int i=0 ; i<(int)num_site-3 ; i++ )
			{
				opfOb.POpChi[i] = prev.projectSopSMatNz( prev.opfOb.POpChi[i] ) ;
			}
			opfOb.POpChi[num_site-3] = prev.projectSopHamil2Nz( prev.opfOb.POpChi[num_site-3] ) ;

			prev.sysHamil.sMatFree() ;

			sysHamil.prodSopRId( sysBlockHamil , num_ssite_state ) ;
			sysBlockHamil.sMatFree() ;

			// For J1 (need whenever)
			sparseMat dumA ,
				  dumC ,
				  dumD ;
			dumA.prodSop( sysBlockSop.Sz , ssiteSop.Sz , J1n * Jz  ) ; sysHamil.addSop3( dumA ) ; dumA.sMatFree() ;
			dumC.prodSop( sysBlockSop.Sp , ssiteSop.Sm , J1n * 0.5 ) ; sysHamil.addSop3( dumC ) ; dumC.sMatFree() ;
			dumD.prodSop( sysBlockSop.Sm , ssiteSop.Sp , J1n * 0.5 ) ; sysHamil.addSop3( dumD ) ; dumD.sMatFree() ;

			// Options for PLocalSS (inserting into the paragraph above)
			//opfOb.POpSS[num_site-2].sMatAllocCpy( dumA );
			//opfOb.POpSS[num_site-2].addSop3( dumA ) ;
			//opfOb.POpSS[num_site-2].addSop3( dumC ) ;
			//opfOb.POpSS[num_site-2].addSop3( dumD ) ;
			//sysHamil.addSop3( opfOb.POpSS[num_site-2] ) ;
					
			// If PLocalSS considered Use freePOpSS()  etc.
			prev.opfOb.freePOpSz () ;
			prev.opfOb.freePOpChi() ;
			prev.opfOb.freePOpSxy() ;
			prev.opfOb.freePOpStrSz() ;
			prev.opfOb.freePOpStrSx() ;
			prev.opfOb.freePOpSstrSz() ;
			prev.opfOb.freePOpSstrSx() ;

			// For J2
			dumA.prodSop( sysBlockSop2.Sz , ssiteSop.Sz , J2*Jz  ) ; sysHamil.addSop3( dumA ) ; dumA.sMatFree() ;
			dumC.prodSop( sysBlockSop2.Sp , ssiteSop.Sm , J2*0.5 ) ; sysHamil.addSop3( dumC ) ; dumC.sMatFree() ;
			dumD.prodSop( sysBlockSop2.Sm , ssiteSop.Sp , J2*0.5 ) ; sysHamil.addSop3( dumD ) ; dumD.sMatFree() ;

			gsl_matrix_free( prev.red_dm_evec ) ;
		}
} ;
