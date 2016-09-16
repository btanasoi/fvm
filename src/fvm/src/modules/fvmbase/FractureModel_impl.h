// This file os part of FVM
// Copyright (c) 2012 FVM Authors
// See LICENSE file for terms.

#include "Mesh.h"
#include <sstream>

#include "NumType.h"
#include "Array.h"
#include "Field.h"
#include "CRConnectivity.h"
#include "LinearSystem.h"
//#include "FieldSet.h"
#include "StorageSite.h"
#include "MultiFieldMatrix.h"
#include "CRMatrix.h"
#include "FluxJacobianMatrix.h"
#include "DiagonalMatrix.h"
#include "GenericBCS.h"
#include "Vector.h"
#include "DiffusionDiscretization.h"
//#include "ConvectionDiscretization.h"
#include "AMG.h"
#include "Linearizer.h"
#include "GradientModel.h"
#include "Underrelaxer.h"
//#include "GenericIBDiscretization.h"
#include "SourceDiscretizationforFracture.h"
#include "TimeDerivativeDiscretization.h"
#include "SquareTensor.h"

template<class T>
class FractureModel<T>::Impl
{
public:
  typedef Array<T> TArray;
  typedef Vector<T,3> VectorT3;
  typedef Array<VectorT3> VectorT3Array;
  typedef Gradient<T> TGradType;
  typedef Array<Gradient<T> > TGradArray;
  typedef CRMatrix<T,T,T> T_Matrix;
  typedef SquareTensor<T,3>  DiagTensorT3;
  
  Impl(const GeomFields& geomFields,
       FractureFields& fractureFields,
       const MeshList& meshes) :
    _meshes(meshes),
    _geomFields(geomFields),
    _fractureFields(fractureFields),
    _phasefieldGradientModel(_meshes,_fractureFields.phasefieldvalue,
                              _fractureFields.phasefieldGradient,_geomFields),
    _initialNorm(),
    _niters(0)
  {
    const int numMeshes = _meshes.size();
    for (int n=0; n<numMeshes; n++)
    {
        const Mesh& mesh = *_meshes[n];
        FractureVC<T> *vc(new FractureVC<T>());
        vc->vcType = "flow";
       _vcMap[mesh.getID()] = vc;
        
        foreach(const FaceGroupPtr fgPtr, mesh.getBoundaryFaceGroups())
        {
            const FaceGroup& fg = *fgPtr;
            FractureBC<T> *bc(new FractureBC<T>());
            
            _bcMap[fg.id] = bc;

            if ((fg.groupType == "wall") ||
                (fg.groupType == "symmetry"))
            {
                bc->bcType = "SpecifiedPhaseFieldFlux";
            }
            else if ((fg.groupType == "velocity-inlet") ||
                     (fg.groupType == "pressure-outlet"))
            {
                bc->bcType = "SpecifiedPhaseFieldValue";
            }
            else
              throw CException("FractureModel: unknown face group type "
                               + fg.groupType);
        }
    }
  }

  void init()
  {
    const int numMeshes = _meshes.size();
    for (int n=0; n<numMeshes; n++)
    {
        const Mesh& mesh = *_meshes[n];

        const StorageSite& cells = mesh.getCells();
        const FractureVC<T>& vc = *_vcMap[mesh.getID()];

	//phasefieldvalue
        shared_ptr<TArray> tCell(new TArray(cells.getCountLevel1()));
        *tCell = _options["initialPhaseFieldValue"];
        _fractureFields.phasefieldvalue.addArray(cells,tCell);
	
	if(_options.transient)
	  {
	    _fractureFields.phasefieldvalueN1.addArray(cells, dynamic_pointer_cast<ArrayBase>(tCell->newCopy()));
	    if (_options.timeDiscretizationOrder > 1)
	      _fractureFields.phasefieldvalueN2.addArray(cells, dynamic_pointer_cast<ArrayBase>(tCell->newCopy()));
	  }

	//conductivity
        shared_ptr<TArray> condCell(new TArray(cells.getCountLevel1()));
        *condCell = vc["fractureConductivity"];
        _fractureFields.conductivity.addArray(cells,condCell);
	
	//source 
	shared_ptr<TArray> sCell(new TArray(cells.getCountLevel1()));
	*sCell = vc["fractureSource"];
	//*sCell =T(0.0);
	_fractureFields.source.addArray(cells,sCell);

	//source coef
	shared_ptr<TArray> scoefCell(new TArray(cells.getCountLevel1()));
	*scoefCell = vc["fractureSourceCoef"];
	//*scoefCell=T(0.0);
	_fractureFields.sourcecoef.addArray(cells,scoefCell);

	//create a zero field
	shared_ptr<TArray> zeroCell(new TArray(cells.getCountLevel1()));
	*zeroCell = T(0.0);
	_fractureFields.zero.addArray(cells,zeroCell);

	//create a one field
	shared_ptr<TArray> oneCell(new TArray(cells.getCountLevel1()));
	*oneCell = T(1.0);
	_fractureFields.one.addArray(cells,oneCell);

	//initial phasefieldvalue gradient array
	shared_ptr<TGradArray> gradT(new TGradArray(cells.getCountLevel1()));
	gradT->zero();
	_fractureFields.phasefieldGradient.addArray(cells,gradT);
        

	//phasefield flux at faces
        foreach(const FaceGroupPtr fgPtr, mesh.getBoundaryFaceGroups())
        {
            const FaceGroup& fg = *fgPtr;
            const StorageSite& faces = fg.site;
          
            shared_ptr<TArray> fluxFace(new TArray(faces.getCount()));

            fluxFace->zero();
            _fractureFields.phasefieldFlux.addArray(faces,fluxFace);
          
        }
        foreach(const FaceGroupPtr fgPtr, mesh.getInterfaceGroups())
        {
            const FaceGroup& fg = *fgPtr;
            const StorageSite& faces = fg.site;
          
            shared_ptr<TArray> fluxFace(new TArray(faces.getCount()));

            fluxFace->zero();
            _fractureFields.phasefieldFlux.addArray(faces,fluxFace);
          
        }
	
	
    }
    _fractureFields.conductivity.syncLocal();
    _niters  =0;
    _initialNorm = MFRPtr();
  }
  
  FractureBCMap& getBCMap() {return _bcMap;}
  FractureVCMap& getVCMap() {return _vcMap;}

  FractureBC<T>& getBC(const int id) {return *_bcMap[id];}

  FractureModelOptions<T>& getOptions() {return _options;}

  void initLinearization(LinearSystem& ls)
  {
    const int numMeshes = _meshes.size();
    for (int n=0; n<numMeshes; n++)
    {
        const Mesh& mesh = *_meshes[n];

        const StorageSite& cells = mesh.getCells();
        MultiField::ArrayIndex tIndex(&_fractureFields.phasefieldvalue,&cells);

        ls.getX().addArray(tIndex,_fractureFields.phasefieldvalue.getArrayPtr(cells));

        const CRConnectivity& cellCells = mesh.getCellCells();

        shared_ptr<Matrix> m(new CRMatrix<T,T,T>(cellCells));

        ls.getMatrix().addMatrix(tIndex,tIndex,m);

        foreach(const FaceGroupPtr fgPtr, mesh.getBoundaryFaceGroups())
        {
            const FaceGroup& fg = *fgPtr;
            const StorageSite& faces = fg.site;

            MultiField::ArrayIndex fIndex(&_fractureFields.phasefieldFlux,&faces);
            ls.getX().addArray(fIndex,_fractureFields.phasefieldFlux.getArrayPtr(faces));

            const CRConnectivity& faceCells = mesh.getFaceCells(faces);

            shared_ptr<Matrix> mft(new FluxJacobianMatrix<T,T>(faceCells));
            ls.getMatrix().addMatrix(fIndex,tIndex,mft);

            shared_ptr<Matrix> mff(new DiagonalMatrix<T,T>(faces.getCount()));
            ls.getMatrix().addMatrix(fIndex,fIndex,mff);
        }

        foreach(const FaceGroupPtr fgPtr, mesh.getInterfaceGroups())
        {
            const FaceGroup& fg = *fgPtr;
            const StorageSite& faces = fg.site;

            MultiField::ArrayIndex fIndex(&_fractureFields.phasefieldFlux,&faces);
            ls.getX().addArray(fIndex,_fractureFields.phasefieldFlux.getArrayPtr(faces));

            const CRConnectivity& faceCells = mesh.getFaceCells(faces);

            shared_ptr<Matrix> mft(new FluxJacobianMatrix<T,T>(faceCells));
            ls.getMatrix().addMatrix(fIndex,tIndex,mft);

            shared_ptr<Matrix> mff(new DiagonalMatrix<T,T>(faces.getCount()));
            ls.getMatrix().addMatrix(fIndex,fIndex,mff);
        }

    }
  }

  void linearize(LinearSystem& ls)
  {
    _phasefieldGradientModel.compute();
    
    DiscrList discretizations;

    shared_ptr<Discretization>
      dd(new DiffusionDiscretization<T,T,T>
	 (_meshes,_geomFields,
	  _fractureFields.phasefieldvalue,
	  _fractureFields.conductivity,
	  _fractureFields.phasefieldGradient));
    discretizations.push_back(dd);
 
    
    shared_ptr<Discretization>
      scfd(new SourceDiscretizationforFracture<T, T, T>
	 (_meshes, 
	  _geomFields, 
	  _fractureFields.phasefieldvalue,
	  _fractureFields.source,
	  _fractureFields.sourcecoef));
    discretizations.push_back(scfd);
    
    if (_options.transient)
      {
	shared_ptr<Discretization>
	  td(new TimeDerivativeDiscretization<T, T, T>
	     (_meshes, _geomFields, 
	      _fractureFields.phasefieldvalue, 
	      _fractureFields.phasefieldvalueN1,
	      _fractureFields.phasefieldvalueN2,
	      _fractureFields.one,
	      _options["timeStep"]));
	discretizations.push_back(td);
      }
    

    Linearizer linearizer;

    linearizer.linearize(discretizations,_meshes,ls.getMatrix(),
                         ls.getX(), ls.getB());

    const int numMeshes = _meshes.size();
    for (int n=0; n<numMeshes; n++)
    {
        const Mesh& mesh = *_meshes[n];

        foreach(const FaceGroupPtr fgPtr, mesh.getBoundaryFaceGroups())
        {
            const FaceGroup& fg = *fgPtr;
            const StorageSite& faces = fg.site;

            const FractureBC<T>& bc = *_bcMap[fg.id];
            

            GenericBCS<T,T,T> gbc(faces,mesh,
                                  _geomFields,
                                  _fractureFields.phasefieldvalue,
                                  _fractureFields.phasefieldFlux,
                                  ls.getMatrix(), ls.getX(), ls.getB());

            if (bc.bcType == "SpecifiedPhaseFieldValue")
            {
	        FloatValEvaluator<T>
                  bT(bc.getVal("specifiedPhaseFieldValue"),faces);
		gbc.applyDirichletBC(bT);
            }
            else if (bc.bcType == "SpecifiedPhaseFieldFlux")
            {
                FloatValEvaluator<T>
                    bPhaseFieldFlux(bc.getVal("specifiedPhaseFieldFlux"),faces);
                    
                const int nFaces = faces.getCount();
                                
                for(int f=0; f<nFaces; f++)
                    {                        
                        gbc.applyNeumannBC(f, bPhaseFieldFlux[f]);
                    }                              
            }
            else if (bc.bcType == "Symmetry")
            {
                 T zeroFlux(NumTypeTraits<T>::getZero());
                 gbc.applyNeumannBC(zeroFlux);
            }
	    else
              throw CException(bc.bcType + " not implemented for FractureModel");
        }

        foreach(const FaceGroupPtr fgPtr, mesh.getInterfaceGroups())
        {
            const FaceGroup& fg = *fgPtr;
            const StorageSite& faces = fg.site;
            GenericBCS<T,T,T> gbc(faces,mesh,
                                  _geomFields,
                                  _fractureFields.phasefieldvalue,
                                  _fractureFields.phasefieldFlux,
                                  ls.getMatrix(), ls.getX(), ls.getB());

            gbc.applyInterfaceBC();
        }
    }
#if 0
    shared_ptr<Discretization>
      ud(new Underrelaxer<T,T,T>
         (_meshes,_fractureFields.phasefieldvalue,
          _options["phasefieldvalueURF"]));
    
    DiscrList discretizations2;
    discretizations2.push_back(ud);

    linearizer.linearize(discretizations2,_meshes,ls.getMatrix(),
                         ls.getX(), ls.getB());
#endif
  }
  
	

  void advance(const int niter)
  {
    for(int n=0; n<niter; n++)
    { 
        LinearSystem ls;
        initLinearization(ls);
        
        ls.initAssembly();

        linearize(ls);

        ls.initSolve();

        MFRPtr rNorm(_options.getLinearSolver().solve(ls));

        if (!_initialNorm) _initialNorm = rNorm;
        
        MFRPtr normRatio((*rNorm)/(*_initialNorm));

#ifdef FVM_PARALLEL	
        if ( MPI::COMM_WORLD.Get_rank() == 0 ){  //only root process
        cout << _niters << ": " << *rNorm << endl;
        }	     
#endif
#ifndef FVM_PARALLEL	
        cout << _niters << ": " << *rNorm << endl;
#endif
        
        _options.getLinearSolver().cleanup();

        ls.postSolve();
        ls.updateSolution();

        _niters++;
        if (*rNorm < _options.absoluteTolerance ||
            *normRatio < _options.relativeTolerance)
          break;
    }
  }
    
  void printBCs()
  {
    foreach(typename FractureBCMap::value_type& pos, _bcMap)
    {
        cout << "Face Group " << pos.first << ":" << endl;
        cout << "    bc type " << pos.second->bcType << endl;
        foreach(typename FractureBC<T>::value_type& vp, *pos.second)
        {
            cout << "   " << vp.first << " "  << vp.second.constant <<  endl;
        }
    }
  }


  void updateTime()
  {
    const int numMeshes = _meshes.size();
    for (int n=0; n<numMeshes; n++)    {
     
      const Mesh& mesh = *_meshes[n];
      const StorageSite& cells = mesh.getCells();
      const int nCells = cells.getCountLevel1();
	
      TArray& phasefieldvalue =
          dynamic_cast<TArray&>(_fractureFields.phasefieldvalue[cells]);
      TArray& phasefieldvalueN1 =
          dynamic_cast<TArray&>(_fractureFields.phasefieldvalueN1[cells]);
     
      if (_options.timeDiscretizationOrder > 1)
        {
	  TArray& phasefieldvalueN2 =
	    dynamic_cast<TArray&>(_fractureFields.phasefieldvalueN2[cells]);
	  phasefieldvalueN2 = phasefieldvalueN1;
        }
      phasefieldvalueN1 = phasefieldvalue;
    }
  }


private:
  const MeshList _meshes;
  const GeomFields& _geomFields;
  FractureFields& _fractureFields;

  FractureBCMap _bcMap;
  FractureVCMap _vcMap;
  FractureModelOptions<T> _options;
  GradientModel<T> _phasefieldGradientModel;
  
  MFRPtr _initialNorm;
  int _niters;
};

template<class T>
FractureModel<T>::FractureModel(const GeomFields& geomFields,
                              FractureFields& fractureFields,
                              const MeshList& meshes) :
  Model(meshes),
  _impl(new Impl(geomFields,fractureFields,meshes))
{
  logCtor();
}


template<class T>
FractureModel<T>::~FractureModel()
{
  logDtor();
}

template<class T>
void
FractureModel<T>::init()
{
  _impl->init();
}
  
template<class T>
typename FractureModel<T>::FractureBCMap&
FractureModel<T>::getBCMap() {return _impl->getBCMap();}

template<class T>
typename FractureModel<T>::FractureVCMap&
FractureModel<T>::getVCMap() {return _impl->getVCMap();}

template<class T>
FractureBC<T>&
FractureModel<T>::getBC(const int id) {return _impl->getBC(id);}

template<class T>
FractureModelOptions<T>&
FractureModel<T>::getOptions() {return _impl->getOptions();}


template<class T>
void
FractureModel<T>::printBCs()
{
  _impl->printBCs();
}

template<class T>
void
FractureModel<T>::advance(const int niter)
{
  _impl->advance(niter);
}

template<class T>
void
FractureModel<T>::updateTime()
{
  _impl->updateTime();
}
