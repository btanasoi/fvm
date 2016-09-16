// This file os part of FVM
// Copyright (c) 2012 FVM Authors
// See LICENSE file for terms.

#ifndef _STRUCTURESOURCEDISCRETIZATION_H_
#define _STRUCTURESOURCEDISCRETIZATION_H_

#include "Field.h"
#include "MultiField.h"
#include "MultiFieldMatrix.h"
#include "Mesh.h"
#include "Discretization.h"
#include "StorageSite.h"
#include "GeomFields.h"
#include "CRConnectivity.h"
#include "CRMatrixRect.h"
#include "Vector.h"
#include "GradientModel.h"

template<class T, class Diag, class OffDiag>
class StructureSourceDiscretization : public Discretization
{
public:

  typedef Array<T> TArray;
  typedef Vector<T,3> VectorT3;
  typedef Array<VectorT3> VectorT3Array;
  typedef Gradient<VectorT3> VGradType;
  typedef Array<Gradient<VectorT3> > VGradArray;

  typedef CRMatrix<Diag,OffDiag,VectorT3> CCMatrix;
  typedef typename CCMatrix::DiagArray DiagArray;
  typedef typename CCMatrix::PairWiseAssembler CCAssembler;

  typedef GradientModel<VectorT3> VGradModelType;
  typedef typename VGradModelType::GradMatrixType VGradMatrix;
  
  StructureSourceDiscretization(const MeshList& meshes,
				const GeomFields& geomFields,
				Field& varField,
				const Field& muField,
				const Field& muoldField,
				const Field& lambdaField,
				const Field& lambdaoldField,
				const Field& alphaField,
				const Field& pfvField,
				const Field& pfperfectField,
				const Field& eigenvalueField,
				const Field& eigenvector1Field,
				const Field& eigenvector2Field,
				const Field& eigenvector3Field,
				const Field& structcoef1Field,
				const Field& structcoef2Field,
				const Field& cvolposField,
				const Field& cvolnegField,
				const Field& cdevField,
				const Field& varGradientField,
				const Field& temperatureField,
                                const T& referenceTemperature,
				const T& residualXXStress,
				const T& residualYYStress,
				const T& residualZZStress,
				const bool& thermo,
				const bool& residualStress,
                                bool fullLinearization=true)  :
    Discretization(meshes),
    _geomFields(geomFields),
    _varField(varField),
    _muField(muField),
    _muoldField(muoldField),
    _lambdaField(lambdaField),
    _lambdaoldField(lambdaoldField),
    _alphaField(alphaField),
    _pfvField(pfvField),
    _pfperfectField(pfperfectField),
    _eigenvalueField(eigenvalueField),
    _eigenvector1Field(eigenvector1Field),
    _eigenvector2Field(eigenvector2Field),
    _eigenvector3Field(eigenvector3Field),
    _structcoef1Field(structcoef1Field),
    _structcoef2Field(structcoef2Field),
	_cvolposField(cvolposField),
	_cvolnegField(cvolnegField),
	_cdevField(cdevField),
    _varGradientField(varGradientField),
    _temperatureField(temperatureField),
    _referenceTemperature(referenceTemperature),
    _residualXXStress(residualXXStress),
    _residualYYStress(residualYYStress),
    _residualZZStress(residualZZStress),
    _thermo(thermo),
    _residualStress(residualStress),
    _fullLinearization(fullLinearization)
   {}

  void discretize(const Mesh& mesh, MultiFieldMatrix& mfmatrix,
                  MultiField& xField, MultiField& rField)
  {
    const StorageSite& iFaces = mesh.getInteriorFaceGroup().site;
    
    discretizeFaces(mesh, iFaces, mfmatrix, xField, rField, false, false);

    /*
    foreach(const FaceGroupPtr fgPtr, mesh.getInterfaceGroups())
    {
        const FaceGroup& fg = *fgPtr;
        const StorageSite& faces = fg.site;
        discretizeFaces(mesh, faces, mfmatrix, xField, rField, false, false);
    }
    */
        
    // boundaries and interfaces
    foreach(const FaceGroupPtr fgPtr, mesh.getAllFaceGroups())
    {
        const FaceGroup& fg = *fgPtr;
        const StorageSite& faces = fg.site;
	if (fg.groupType!="interior")
	{
	    discretizeFaces(mesh, faces, mfmatrix, xField, rField,
			    fg.groupType!="interface",
			    fg.groupType=="symmetry");
	}
    }
  }

                          
  void discretizeFaces(const Mesh& mesh, const StorageSite& faces,
                       MultiFieldMatrix& mfmatrix,
                       MultiField& xField, MultiField& rField,
                       const bool isBoundary, const bool isSymmetry)
  {
    const StorageSite& cells = mesh.getCells();

    const MultiField::ArrayIndex cVarIndex(&_varField,&cells);

    const VectorT3Array& faceArea =
      dynamic_cast<const VectorT3Array&>(_geomFields.area[faces]);

    const TArray& faceAreaMag =
      dynamic_cast<const TArray&>(_geomFields.areaMag[faces]);

    const VectorT3Array& cellCentroid =
      dynamic_cast<const VectorT3Array&>(_geomFields.coordinate[cells]);

    const VectorT3Array& faceCentroid =
      dynamic_cast<const VectorT3Array&>(_geomFields.coordinate[faces]);


    const TArray& cellVolume =
      dynamic_cast<const TArray&>(_geomFields.volume[cells]);

    const CRConnectivity& faceCells = mesh.getFaceCells(faces);

    VectorT3Array& rCell = dynamic_cast<VectorT3Array&>(rField[cVarIndex]);
    const VectorT3Array& xCell = dynamic_cast<const VectorT3Array&>(xField[cVarIndex]);

    const VGradArray& vGradCell =
      dynamic_cast<const VGradArray&>(_varGradientField[cells]);

    const TArray& muCell =
      dynamic_cast<const TArray&>(_muField[cells]);

    const TArray& muoldCell =
      dynamic_cast<const TArray&>(_muoldField[cells]);

    const TArray& lambdaCell =
      dynamic_cast<const TArray&>(_lambdaField[cells]);
      
    const TArray& lambdaoldCell =
      dynamic_cast<const TArray&>(_lambdaoldField[cells]);

    const TArray& alphaCell =
      dynamic_cast<const TArray&>(_alphaField[cells]);
      
    const TArray& pfvCell =
      dynamic_cast<const TArray&>(_pfvField[cells]);
      
    const TArray& pfperfectCell =
      dynamic_cast<const TArray&>(_pfperfectField[cells]);
      
    const VectorT3Array& eigenvalueCell =
      dynamic_cast<const VectorT3Array&>(_eigenvalueField[cells]);
    
    const VectorT3Array& eigenvector1Cell =
      dynamic_cast<const VectorT3Array&>(_eigenvector1Field[cells]);
    const VectorT3Array& eigenvector2Cell =
      dynamic_cast<const VectorT3Array&>(_eigenvector2Field[cells]);
    const VectorT3Array& eigenvector3Cell =
      dynamic_cast<const VectorT3Array&>(_eigenvector3Field[cells]);
      
    const TArray& structcoef1Cell =
      dynamic_cast<const TArray&>(_structcoef1Field[cells]);

    const TArray& structcoef2Cell =
      dynamic_cast<const TArray&>(_structcoef2Field[cells]);

	const TArray& cvolposCell =
      dynamic_cast<const TArray&>(_cvolposField[cells]);

	const TArray& cvolnegCell =
      dynamic_cast<const TArray&>(_cvolnegField[cells]);

	const TArray& cdevCell =
      dynamic_cast<const TArray&>(_cdevField[cells]);
	  
    const TArray& temperatureCell =
      dynamic_cast<const TArray&>(_temperatureField[cells]);

    CCMatrix& matrix = dynamic_cast<CCMatrix&>(mfmatrix.getMatrix(cVarIndex,
                                                             cVarIndex));
    CCAssembler& assembler = matrix.getPairWiseAssembler(faceCells);
    DiagArray& diag = matrix.getDiag();


    const int nFaces = faces.getCount();

    const VGradMatrix& vgMatrix = VGradModelType::getGradientMatrix(mesh,_geomFields);
    const CRConnectivity& cellCells = vgMatrix.getConnectivity();
    const Array<int>& ccRow = cellCells.getRow();
    const Array<int>& ccCol = cellCells.getCol();

    const int nInteriorCells = cells.getSelfCount();

    const T two(2.0);
    const T three(3.0);
    
    for(int f=0; f<nFaces; f++)
    {
        const int c0 = faceCells(f,0);
        const int c1 = faceCells(f,1);

	const VectorT3& Af = faceArea[f];
        const VectorT3 en = Af/faceAreaMag[f];
        
        VectorT3 ds=cellCentroid[c1]-cellCentroid[c0];

        T vol0 = cellVolume[c0];
        T vol1 = cellVolume[c1];

        T wt0 = vol0/(vol0+vol1);
        T wt1 = vol1/(vol0+vol1);

        if (isBoundary && !isSymmetry)
        {  
            wt0 = T(1.0);
            wt1 = T(0.);
        }
        
        T faceMu(1.0);
        T faceMuOld(1.0);
	    T faceLambda(1.0);
	    T faceLambdaOld(1.0);
        T faceAlpha(1.0);
        T faceTemperature(1.0);
        T faceStructcoef1(1.0);
        T faceStructcoef2(1.0);
		T facecvolpos(1.0);
		T facecvolneg(1.0);
		T facecdev(1.0);
        
        VectorT3 faceEigenvalue11;
        VectorT3 faceEigenvalue12;
        VectorT3 faceEigenvalue13;
        VectorT3 faceEigenvalue21;
        VectorT3 faceEigenvalue22;
        VectorT3 faceEigenvalue23;
        VectorT3 faceEigenvalue31;
        VectorT3 faceEigenvalue32;
        VectorT3 faceEigenvalue33;

        Diag& a00 = diag[c0];
        Diag& a11 = diag[c1];
        OffDiag& a01 = assembler.getCoeff01(f);
        OffDiag& a10 = assembler.getCoeff10(f);

        if (vol0 == 0.)
       	{
            faceMu = muCell[c1];
	    faceLambda = lambdaCell[c1];
	    faceAlpha = alphaCell[c1];
            faceTemperature = temperatureCell[c1];}
        else if (vol1 == 0.)
        {
            faceMu = muCell[c0];
	    faceLambda = lambdaCell[c0];
	    faceAlpha = alphaCell[c0];
            faceTemperature = temperatureCell[c0];}
        else
        {
            faceMu = harmonicAverage(muCell[c0],muCell[c1]);
	    faceLambda = harmonicAverage(lambdaCell[c0],lambdaCell[c1]);
	    faceAlpha = harmonicAverage(alphaCell[c0],alphaCell[c1]);
            faceTemperature = harmonicAverage(temperatureCell[c0],temperatureCell[c1]);}

        faceMu = muCell[c0]*wt0 + muCell[c1]*wt1;
        faceMuOld = muoldCell[c0]*wt0 + muoldCell[c1]*wt1;
        faceLambda = lambdaCell[c0]*wt0 + lambdaCell[c1]*wt1;
        faceLambdaOld = lambdaoldCell[c0]*wt0 + lambdaoldCell[c1]*wt1;
        faceAlpha = alphaCell[c0]*wt0 + alphaCell[c1]*wt1;
        faceTemperature = temperatureCell[c0]*wt0 + temperatureCell[c1]*wt1;
        faceStructcoef1 = structcoef1Cell[c0]*wt0 + structcoef1Cell[c1]*wt1;
        faceStructcoef2 = structcoef2Cell[c0]*wt0 + structcoef2Cell[c1]*wt1;
		facecvolpos = cvolposCell[c0]*wt0 + cvolposCell[c1]*wt1;
		facecvolneg = cvolnegCell[c0]*wt0 + cvolnegCell[c1]*wt1;
		facecdev = cdevCell[c0]*wt0 + cdevCell[c1]*wt1;
        
               
    const VGradType gradF = (vGradCell[c0]*wt0 + vGradCell[c1]*wt1);

	VectorT3 source(NumTypeTraits<VectorT3>::getZero());
	VectorT3 thermalSource(NumTypeTraits<VectorT3>::getZero());
	VectorT3 residualSource(NumTypeTraits<VectorT3>::getZero());
	
        const T divU = (gradF[0][0] + gradF[1][1] + gradF[2][2]);
        const T diffMetric = faceAreaMag[f]*faceAreaMag[f]/dot(faceArea[f],ds);
        const VectorT3 secondaryCoeff = faceMu*(faceArea[f]-ds*diffMetric);
        
        
        const T divUc0 = (eigenvector1Cell[c0][0] +eigenvector2Cell[c0][1] +eigenvector3Cell[c0][2]);
        const T divUc1 = (eigenvector1Cell[c1][0] +eigenvector2Cell[c1][1] +eigenvector3Cell[c1][2]);
        
        faceEigenvalue11[0]=2.0*(1.0-pfvCell[c0]*pfvCell[c0])*(eigenvector1Cell[c0][0]-divUc0/3.0)*wt0 + 2.0*(1.0-pfvCell[c1]*pfvCell[c1])*(eigenvector1Cell[c1][0]-divUc1/3.0)*wt1;
        faceEigenvalue12[0]=2.0*(1.0-pfvCell[c0]*pfvCell[c0])*eigenvector1Cell[c0][1]*wt0 + 2.0*(1.0-pfvCell[c1]*pfvCell[c1])*eigenvector1Cell[c1][1]*wt1;
        faceEigenvalue13[0]=2.0*(1.0-pfvCell[c0]*pfvCell[c0])*eigenvector1Cell[c0][2]*wt0 + 2.0*(1.0-pfvCell[c1]*pfvCell[c1])*eigenvector1Cell[c1][2]*wt1;
        
        faceEigenvalue21[0]=2.0*(1.0-pfvCell[c0]*pfvCell[c0])*eigenvector2Cell[c0][0]*wt0 + 2.0*(1.0-pfvCell[c1]*pfvCell[c1])*eigenvector2Cell[c1][0]*wt1;
        faceEigenvalue22[0]=2.0*(1.0-pfvCell[c0]*pfvCell[c0])*(eigenvector2Cell[c0][1]-divUc0/3.0)*wt0 + 2.0*(1.0-pfvCell[c1]*pfvCell[c1])*(eigenvector2Cell[c1][1]-divUc1/3.0)*wt1;
        faceEigenvalue23[0]=2.0*(1.0-pfvCell[c0]*pfvCell[c0])*eigenvector2Cell[c0][2]*wt0 + 2.0*(1.0-pfvCell[c1]*pfvCell[c1])*eigenvector2Cell[c1][2]*wt1;
        
        faceEigenvalue31[0]=2.0*(1.0-pfvCell[c0]*pfvCell[c0])*eigenvector3Cell[c0][0]*wt0 + 2.0*(1.0-pfvCell[c1]*pfvCell[c1])*eigenvector3Cell[c1][0]*wt1;
        faceEigenvalue32[0]=2.0*(1.0-pfvCell[c0]*pfvCell[c0])*eigenvector3Cell[c0][1]*wt0 + 2.0*(1.0-pfvCell[c1]*pfvCell[c1])*eigenvector3Cell[c1][1]*wt1;
        faceEigenvalue33[0]=2.0*(1.0-pfvCell[c0]*pfvCell[c0])*(eigenvector3Cell[c0][2]-divUc0/3.0)*wt0 + 2.0*(1.0-pfvCell[c1]*pfvCell[c1])*(eigenvector3Cell[c1][2]-divUc1/3.0)*wt1;

        // mu*grad U ^ T + lambda * div U I
	source[0] = faceMu*(gradF[0][0]*Af[0] + gradF[0][1]*Af[1] + gradF[0][2]*Af[2])
          + faceLambda*divU*Af[0];
        
	source[1] = faceMu*(gradF[1][0]*Af[0] + gradF[1][1]*Af[1] + gradF[1][2]*Af[2])
          + faceLambda*divU*Af[1];
        
	source[2] = faceMu*(gradF[2][0]*Af[0] + gradF[2][1]*Af[1] + gradF[2][2]*Af[2])
          + faceLambda*divU*Af[2];
      
    
    source[0] -= facecdev*faceMuOld*(faceEigenvalue11[0]*Af[0] +faceEigenvalue21[0]*Af[1] + faceEigenvalue31[0]*Af[2]);
    source[1] -= facecdev*faceMuOld*(faceEigenvalue12[0]*Af[0] +faceEigenvalue22[0]*Af[1] + faceEigenvalue32[0]*Af[2]);
    source[2] -= facecdev*faceMuOld*(faceEigenvalue13[0]*Af[0] +faceEigenvalue23[0]*Af[1] + faceEigenvalue33[0]*Af[2]);
    
    //printf("source term: %lf, %lf, %lf\n",source[0],source[1],source[2]);
    
    if (divU>0 && (pfperfectCell[c0]!=-1&&pfperfectCell[c1]!=-1)){
        source[0] -= facecvolpos*((1.0-pfvCell[c0]*pfvCell[c0])*wt0+(1.0-pfvCell[c1]*pfvCell[c1])*wt1)*(faceLambdaOld+2.0/3.0*faceMuOld)*divU*Af[0];
        source[1] -= facecvolpos*((1.0-pfvCell[c0]*pfvCell[c0])*wt0+(1.0-pfvCell[c1]*pfvCell[c1])*wt1)*(faceLambdaOld+2.0/3.0*faceMuOld)*divU*Af[1];
        source[2] -= facecvolpos*((1.0-pfvCell[c0]*pfvCell[c0])*wt0+(1.0-pfvCell[c1]*pfvCell[c1])*wt1)*(faceLambdaOld+2.0/3.0*faceMuOld)*divU*Af[2];
    }
    if (divU<0 && (pfperfectCell[c0]!=-1&&pfperfectCell[c1]!=-1)){
        source[0] -= facecvolneg*((1.0-pfvCell[c0]*pfvCell[c0])*wt0+(1.0-pfvCell[c1]*pfvCell[c1])*wt1)*(faceLambdaOld+2.0/3.0*faceMuOld)*divU*Af[0];
        source[1] -= facecvolneg*((1.0-pfvCell[c0]*pfvCell[c0])*wt0+(1.0-pfvCell[c1]*pfvCell[c1])*wt1)*(faceLambdaOld+2.0/3.0*faceMuOld)*divU*Af[1];
        source[2] -= facecvolneg*((1.0-pfvCell[c0]*pfvCell[c0])*wt0+(1.0-pfvCell[c1]*pfvCell[c1])*wt1)*(faceLambdaOld+2.0/3.0*faceMuOld)*divU*Af[2];
    }

    if (pfperfectCell[c0]==-1||pfperfectCell[c1]==-1){
        source[0] -= ((1.0-pfvCell[c0]*pfvCell[c0])*wt0+(1.0-pfvCell[c1]*pfvCell[c1])*wt1)*(faceLambdaOld+2.0/3.0*faceMuOld)*divU*Af[0];
        source[1] -= ((1.0-pfvCell[c0]*pfvCell[c0])*wt0+(1.0-pfvCell[c1]*pfvCell[c1])*wt1)*(faceLambdaOld+2.0/3.0*faceMuOld)*divU*Af[1];
        source[2] -= ((1.0-pfvCell[c0]*pfvCell[c0])*wt0+(1.0-pfvCell[c1]*pfvCell[c1])*wt1)*(faceLambdaOld+2.0/3.0*faceMuOld)*divU*Af[2];
    }

	if(_thermo)
	{
	    if(mesh.getDimension()==2)
	      thermalSource -= (two*faceLambda+two*faceMu)*faceAlpha*(faceTemperature-_referenceTemperature)*Af;
	    else
	      thermalSource -= (three*faceLambda+two*faceMu)*faceAlpha*(faceTemperature-_referenceTemperature)*Af;
	}

	if(_residualStress)
	{
	    residualSource[0] = _residualXXStress*Af[0];
	    residualSource[1] = _residualYYStress*Af[1];
	    residualSource[2] = _residualZZStress*Af[2];
	}
        
        VectorT3 s0(NumTypeTraits<VectorT3>::getZero());
        VectorT3 s1(NumTypeTraits<VectorT3>::getZero());
        if (_fullLinearization)
        {
            // loop over level 0 neighbors
            for(int nnb = ccRow[c0]; nnb<ccRow[c0+1]; nnb++)
            {
                const int nb = ccCol[nnb];

                // get the coefficient from the gradient matrix that uses modified cellCells
                // getting a copy rather than reference since we might modify it
                VectorT3 g_nb = vgMatrix.getCoeff(c0,nb);
#if 1
                if (isSymmetry)
                {
                    const T gnb_dot_nx2 = T(2.0)*dot(en,g_nb);
                    g_nb = gnb_dot_nx2*en;
                }
#endif
                Diag coeff;

                for(int i=0; i<3; i++)
                {
                    for(int j=0; j<3; j++)
                    {
                        coeff(i,j) = wt0*(faceMu*Af[j]*g_nb[i] 
                                          + faceLambda*Af[i]*g_nb[j]
                                          );
                    }

                    for(int k=0; k<3; k++)
                      coeff(i,i) += wt0*secondaryCoeff[k]*g_nb[k];
                }
#if 0
                if (isSymmetry)
                {
                    SquareTensor<T,3> R;
                    
                    for(int i=0;i<3;i++)
                      for(int j=0; j<3; j++)
                      {
                          if (i==j)
                            R(i,j) = 1.0 - 2*en[i]*en[j];
                          else
                            R(i,j) = - 2*en[i]*en[j];
                      }

                    Diag coeff1(R*coeff*R);
                    coeff += coeff1;
                      
                }
#endif
                OffDiag& a0_nb = matrix.getCoeff(c0,nb);

                a0_nb += coeff;
                a00 -= coeff;
                a10 += coeff;
                
                if (c1 != nb)
                {
                    // if the coefficient does not exist we assume c1
                    // is a ghost cell for which we don't want an
                    // equation in this matrix
                    if (matrix.hasCoeff(c1,nb))
                    {
                        OffDiag& a1_nb = matrix.getCoeff(c1,nb);
                        a1_nb -= coeff;
                    }
                }
                else
                  a11 -= coeff;
            }


            if (!isBoundary)
            {
                for(int nnb = ccRow[c1]; nnb<ccRow[c1+1]; nnb++)
                {
                    const int nb = ccCol[nnb];
                    const VectorT3& g_nb = vgMatrix.getCoeff(c1,nb);
                    
                    Diag coeff;
                    
                    for(int i=0; i<3; i++)
                    {
                        for(int j=0; j<3; j++)
                        {
                            coeff(i,j) = wt1*(faceMu*Af[j]*g_nb[i] 
                                              + faceLambda*Af[i]*g_nb[j]
                                              );
                        }
                        
                        for(int k=0; k<3; k++)
                          coeff(i,i) += wt1*secondaryCoeff[k]*g_nb[k];
                    }
                    
                    
                    if (matrix.hasCoeff(c1,nb))
                    {
                        OffDiag& a1_nb = matrix.getCoeff(c1,nb);
                        a1_nb -= coeff;
                        a11 += coeff;
                    }
                    a01 -= coeff;
                    
                    
                    if (c0 != nb)
                    {
                        OffDiag& a0_nb = matrix.getCoeff(c0,nb);
                        
                        a0_nb += coeff;
                    }
                    else
                      a00 += coeff;
                    
                }
            }
        }
        
        // mu*gradU, primary part
        

        source += faceMu*diffMetric*(xCell[c1]-xCell[c0]);

        // mu*gradU, secondart part


        source += gradF*secondaryCoeff;

        // add flux to the residual of c0 and c1
        rCell[c0] += source;
	rCell[c1] -= source;

        // add flux due to thermal Stress to the residual of c0 and c1
        rCell[c0] += thermalSource;
        rCell[c1] -= thermalSource;

	// add flux due to residual Stress to the residual of c0 and c1
	rCell[c0] += residualSource;
	rCell[c1] -= residualSource;
     

        // for Jacobian, use 2*mu + lambda as the diffusivity
        const T faceDiffusivity = faceMu;
        const T diffCoeff = faceDiffusivity*diffMetric;

        
        a01 +=diffCoeff;
        a10 +=diffCoeff;

        
        a00 -= diffCoeff;
        a11 -= diffCoeff;

    }
  }
    

private:
  const GeomFields& _geomFields;
  Field& _varField;
  const Field& _muField;
  const Field& _muoldField;
  const Field& _lambdaField;
  const Field& _lambdaoldField;
  const Field& _alphaField;
  const Field& _pfvField;
  const Field& _pfperfectField;
  const Field& _eigenvalueField;
  const Field& _eigenvector1Field;
  const Field& _eigenvector2Field;
  const Field& _eigenvector3Field;
  const Field& _structcoef1Field;
  const Field& _structcoef2Field;
  const Field& _cdevField;
  const Field& _cvolposField;
  const Field& _cvolnegField;
  const Field& _varGradientField;
  const Field& _temperatureField;
  const T _referenceTemperature;
  const T _residualXXStress;
  const T _residualYYStress;
  const T _residualZZStress;
  const bool _thermo;
  const bool _residualStress;
  const bool _fullLinearization; 
};

#endif
