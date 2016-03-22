#!/usr/bin/env python
import fvm
import fvm.fvmbaseExt as fvmbaseExt
import fvm.importers as importers
import fvm.fvmparallel as fvmparallel
import fvm.models_atyped_double as models
import fvm.exporters_atyped_double as exporters
from fvm.fvmbaseExt import VecD3
fvm.set_atype('double')

import time
from numpy import *
from mpi4py import MPI

import pdb
import copy
import sys
from Tools import *
from ComputeForce import *
from optparse import OptionParser
from FluentCase import FluentCase

import tecplotEntireStructureDomainPara
import tecplotEntireFractureDomainPara

##########################################################################################   
# parameter set up
 ########################################################################################## 
#homoCase-10000
beamTop = 6
beamBot = 7
beamLeft = 8
beamRight = 3
beamBack = 5 
beamFront = 4
IDs = (3,4,5,6,7,8)
BoundaryPositionTop = 1e-3-1e-5 #Top
BoundaryPositionRight = 1e-3-1e-5 #Right
BoundaryPositionLeft = 1e-5
BoundaryPositionBottom = 1e-5

# Set Parameters
numSteps = 1			       # Number of Steps
timeStep = 3600                # Size of timestep (seconds)
numtimeSteps = 1               # Number of timesteps in global combined solution
numStructIterations = 20       # Number of iterations for structure model, automatically set to 1 when StructIterFlag == 0
numPFIterations = 10           # Number of iterations for fracture model
E = 1e9                    # Matrix Young's Modulus
E_fiber=19.5e9                 # Fiber Young's Modulus
nu = 0.3                      # Matrix Poisson's ratio
nu_fiber=0.28                  # Fiber Poisson's ratio
G = E/2.0/(1+nu)               # Shear Modulus
K = E/3.0/(1-2.0*nu)
Lamda = nu*E/(1+nu)/(1-2.0*nu) # Lamda
Ac =0                          # Creep coefficient 1/hr
cFED = 1                    # critical fracture energy density, J/m^2
cLoC=  1e-5                    # model parameter controlling the width of the smooth approximation of the crack, m
Diff = 4.0*cLoC*cLoC           # fracture Conductivity Coefficient
crackPF = 1e-3			       # Phase Field Value at Crack
NumofFiber = 0
alpha = 70e-6

DeformUnit = (cFED*BoundaryPositionTop/Lamda)**0.5  #Normalized Displacement Unit
#DispStep = 0.02*DeformUnit	   # Displacement Step
DispStep = 1e-6
StressStep = -1e6
LoadCoef = -0.5
TempStep = 10

OInterval_s = 1                 #Output interval for equilibrium status
OInterval_l = 50
MidOInterval_s = 5              #Output interval for intermediate status
MidOInterval_l = 50
OPFLimit = 0.02
OUpLimit=0                   #Upper Limit for large displacement step
DispReFactor=1.0               #Smaller displacement step is: 1/DispReFactor of larger displacement step
MidIterUpLimit = 200

StiffnessResidual = 1e-6       #Used to have a lower bound of the material constant for damaged cell
StructTolerance = 1e-3         #Tolerance for structure model inner iteration
StructOuterTolerance = 1e-3
StructIterFlag = 0             #1--Do structure model iteration; 0--No structure model iteration

PFTolerance = 1e-5             #Tolerance for fracture model iteration
PFOuterTolerance = 1e-5
PFIterFlag = 1                 #1--Do convergence test iteration; 0--No convergence test iteration

PerfectRad = 0
SymFlag = 1 # 1--Symmetric 0--Asymmetric
trace_change_threshold = 1e-8

structure_file_name = "tecplotresult_structure-pmma-fiber" + ".dat"
inter_status_file_name = "tecplotresult_inter-pmma-fiber" + ".dat"
equil_status_file_name = "tecplotresult_equil-pmma-fiber" + ".dat"
fracture_file_name = "tecplotresult_fracture-pmma-fiber" + ".dat" 

fiber_file_name = "fiber"+".txt"  

##########################################################################################
#End of parameter set up
##########################################################################################
tectype = {
        'tri' : 'FETRIANGLE',
        'quad' : 'FEQUADRILATERAL',
        'tetra' : 'FETETRAHEDRON',
        'hexa' : 'FEBRICK'
        }
# Define the mesh
etype = {
        'tri' : 1,
        'quad' : 2,
        'tetra' : 3,
        'hexa' : 4
        }
parser = OptionParser()
parser.set_defaults(type='quad')
parser.add_option("--type", help="'tri', 'quad'[default], 'hexa', or 'tetra'")
(options, args) = parser.parse_args()
reader = FluentCase(args[0])
reader.read()
t0 = time.time()
fluent_meshes = reader.getMeshList()
nmesh = 1
npart = [MPI.COMM_WORLD.Get_size()]
etype = [etype[options.type]]
if not MPI.COMM_WORLD.Get_rank():
   print "parmesh is processing"
part_mesh = fvmparallel.MeshPartitioner( fluent_meshes, npart, etype );
part_mesh.setWeightType(0);
part_mesh.setNumFlag(0);
part_mesh.partition()
part_mesh.mesh()
meshes = part_mesh.meshList()

geomFields =  models.GeomFields('geom')
globalMetricsCalculator = models.MeshMetricsCalculatorA(geomFields,fluent_meshes)
globalMetricsCalculator.init()
localMetricsCalculator = models.MeshMetricsCalculatorA(geomFields,meshes)
localMetricsCalculator.init()

# Define the fracture and structure models
fractureFields =  models.FractureFields('fracture')
tmodel = models.FractureModelA(geomFields,fractureFields,meshes)
structureFields =  models.StructureFields('structure')
smodel = models.StructureModelA(geomFields, structureFields, meshes)
##########################################################################################
#Set up the boundary conditions
##########################################################################################
FracturebcMap = tmodel.getBCMap()
#for id in PF_VB:
#    if id in FracturebcMap:
#       bc = tmodel.getBCMap()[id]
#       bc.bcType = 'SpecifiedPhaseFieldValue'
#       bc.setVar('specifiedPhaseFieldValue',1.0)       
#       #bc.bcType = 'SpecifiedPhaseFieldFlux'
#       #bc.setVar('specifiedPhaseFieldFlux',0)
#for id in PF_FB:
for id in IDs:
    if id in FracturebcMap:
       bc = tmodel.getBCMap()[id]
       #bc.bcType = 'SpecifiedPhaseFieldValue'
       #bc.setVar('specifiedPhaseFieldValue',1.0)       
       bc.bcType = 'SpecifiedPhaseFieldFlux'
       bc.setVar('specifiedPhaseFieldFlux',0)

vcMap = tmodel.getVCMap()
for vc in vcMap.values():
    #print vc
    vc.setVar('fractureSource',0.0)
    vc.setVar('fractureSourceCoef',0.0)
    vc.setVar('fractureConductivity',Diff)

StructurebcMap = smodel.getBCMap()
for id in [beamRight]:
    if id in StructurebcMap:
        bc = StructurebcMap[id]
        #bc.bcType = 'Symmetry'
        bc.bcType = 'SpecifiedTraction'
        bc['specifiedXXTraction'] = 0
        bc['specifiedYXTraction'] = 0
        bc['specifiedZXTraction'] = 0
for id in [beamTop]:
    if id in StructurebcMap:
        bc = StructurebcMap[id]
        #bc.bcType = 'SpecifiedDeformation'
        #bc['specifiedXDeformation'] = 0
        #bc['specifiedYDeformation'] = 0
        #bc['specifiedZDeformation'] = 0
        bc.bcType = 'SpecifiedTraction'
        bc['specifiedXYTraction'] = 0
        bc['specifiedYYTraction'] = 0
        bc['specifiedZYTraction'] = 0
for id in [beamBot]:
    if id in StructurebcMap:
        bc = StructurebcMap[id]
        #bc.bcType = 'SpecifiedTraction'
        #bc['specifiedXXTraction'] = 0
        #bc['specifiedYXTraction'] = 0
        #bc['specifiedZXTraction'] = 0
        bc.bcType = 'Symmetry'
        #bc.bcType = 'SpecifiedDeformation'
        #bc['specifiedXDeformation'] = 0
        #bc['specifiedYDeformation'] = 0
        #bc['specifiedZDeformation'] = 0
for id in [beamLeft]:
    if id in StructurebcMap:
        bc = StructurebcMap[id]
        bc.bcType = 'Symmetry'
        #bc.bcType = 'SpecifiedTraction'
        #bc['specifiedXZTraction'] = 0
        #bc['specifiedYZTraction'] = 0
        #bc['specifiedZZTraction'] = 0
for id in [beamBack]:
    if id in StructurebcMap:
        bc = StructurebcMap[id]
        bc.bcType = 'Symmetry'
        #bc.bcType = 'SpecifiedTraction'
        #bc['specifiedXZTraction'] = 0
        #bc['specifiedYZTraction'] = 0
        #bc['specifiedZZTraction'] = 0
for id in [beamFront]:
    if id in StructurebcMap:
        bc = StructurebcMap[id]
        #bc.bcType = 'Symmetry'
        bc.bcType = 'SpecifiedTraction'
        bc['specifiedXZTraction'] = 0
        bc['specifiedYZTraction'] = 0
        bc['specifiedZZTraction'] = 0


vcMap = smodel.getVCMap()
for i,vc in vcMap.iteritems():
    vc['density'] = 8912
    vc['eta'] = E/(2.*(1+nu))
    vc['eta1'] = nu*E/((1+nu)*(1-2.0*nu))
    vc['alpha'] = alpha
##########################################################################################
#End of the boundary conditions set up
##########################################################################################
# Define the equation sol. method for the fracture problem
tSolver = fvmbaseExt.AMG()
#tSolver.smootherType = fvmbaseExt.AMG.JACOBI
tSolver.relativeTolerance = 1e-9
tSolver.nMaxIterations = 200000
tSolver.maxCoarseLevels=20
tSolver.verbosity=0
#tSolver.setMergeLevelSize(40000)
# Set vacany model options
toptions = tmodel.getOptions()
toptions.linearSolver = tSolver
toptions.setVar("initialPhaseFieldValue", 1.0)
#toptions.setVar("phasefieldvalueURF", 1.0)
# Initialize the fracture model
tmodel.init()
# Define the equation sol. method for the Structure problem
if StructIterFlag==1:
    defSolver = fvmbaseExt.AMG()
else :
    defSolver = fvmbaseExt.DirectSolver()
    #defSolver.smootherType = fvmbaseExt.AMG.JACOBI
    #soptions.setVar("deformationURF",0.0001)
#defSolver.preconditioner = pc
#defSolver.relativeTolerance = 1e-10
#defSolver.absoluteTolerance = 1.e-20
#defSolver.nMaxIterations = 1000
#defSolver.maxCoarseLevels=20
defSolver.verbosity=0
#defSolver.cycleType = fvmbaseExt.AMG.W_CYCLE
#defSolver.nPreSweeps  = 4
#defSolver.nPostSweeps = 1
# Set structure model options
soptions = smodel.getOptions()
soptions.deformationLinearSolver = defSolver
#soptions.deformationTolerance=1.0e-3
#soptions.printNormalizedResiduals=True
soptions.transient = True
soptions.timeDiscretizationOrder = 2
soptions.setVar("timeStep", timeStep)
#soptions.setVar("deformationURF", 1.0)
soptions.creep = True
soptions.A = Ac/3600
soptions.B = 0
soptions.m = 1
soptions.n = 0.5
soptions.Sy0 = 1e9
# Initialize the structure model
smodel.init()
#Set back to False
soptions.transient = False
soptions.creep = False
soptions.thermo = True
soptions.residualStress = True
##########################################################################################
# Model Initialization
##########################################################################################
rank_id=MPI.COMM_WORLD.Get_rank()
if rank_id == 0:
   f_structure = open(structure_file_name, 'w')
   f_structure.close()
   ss_structure = open(inter_status_file_name,'w')
   sp_structure = open(equil_status_file_name,'w') 
   f_fracture = open(fracture_file_name, 'w')
   f_fracture.close()

if NumofFiber!=0:
    f_fiber = open( fiber_file_name, "r" )
    fiber_r = []
    fiber_x = []
    fiber_y = []
    fiber_count = 1
    for line in f_fiber:
        if(fiber_count % 3 ==1):
            fiber_r.append(float(line))
        if(fiber_count % 3 ==2):
            fiber_x.append(float(line))
        if(fiber_count % 3 ==0):
            fiber_y.append(float(line))
        fiber_count=fiber_count+1
    for i in range(0,NumofFiber):
        print fiber_r[i],fiber_x[i],fiber_y[i]
    f_fiber.close()

EnergyHistoryField = []
PFHistoryField = []
PFPerfectField = []
DeformationHistoryX = []
DeformationHistoryY = []
DeformationHistoryZ = []
V_flag = []
deformation_x_inner = []
deformation_y_inner = []
deformation_z_inner = []
deformation_x_outer = []
deformation_y_outer = []
deformation_z_outer = []
PF_stored = []
PF_inner = []
cellSitesLocal = []
E_local = []
nu_local = []
K_local = []
Lamda_local = []
G_local = []

strain_trace = []
ElasticEnergyField = []

compress_found_flag  = array([0.0])
struct_outer_tol_flag = array([0.0])
struct_outer_flag = array([0.0])
struct_inner_flag = array([0.0])
fract_inner_flag = array([0.0])
mid_loop_flag = array([0.0])
PF_min = array([1.0])
PF_change_max= array([0.0])
PF_change_max_inner= array([0.0])
deformation_change_max = array([0.0])
LoadingTop  = array([0.0])
LoadingRight  = array([0.0])
LoadingBottom  = array([0.0])
LoadingLeft  = array([0.0])

Local_MVStress=0
Local_MDStress=0
Local_MVStrain=0
Local_MDStrain=0

for n in range(0,nmesh):
    cellSitesLocal.append(meshes[n].getCells() )
    Count = cellSitesLocal[n].getCount()
    selfCount = cellSitesLocal[n].getSelfCount()
    coord=geomFields.coordinate[cellSitesLocal[n]]
    coordA=coord.asNumPyArray()
    volume=geomFields.volume[cellSitesLocal[n]]
    volumeA=volume.asNumPyArray()
    etaFields = structureFields.eta[cellSitesLocal[n]]
    etaFieldsA = etaFields.asNumPyArray() 
    eta1Fields = structureFields.eta1[cellSitesLocal[n]]
    eta1FieldsA = eta1Fields.asNumPyArray()

    temperatureFields = structureFields.temperature[cellSitesLocal[n]]
    temperatureFieldsA = temperatureFields.asNumPyArray()

    for i in range(0,Count):
################Pre-defined crack#####################
        PFHistoryField.append(1.0)
        #if (coordA[i,0]-0.0)>0.0 and\
        #(coordA[i,0]-0.1/2.0)<0.0 and\
        #(coordA[i,1]-0.04/2.0+1e-4)>0.0 and\
        #(coordA[i,1]-0.04/2.0-1e-4)<0.0:
        #    PFHistoryField[i]=0   
################Forcing perfect region################  
        PFPerfectField.append(0.0)
        if (coordA[i,1]-0.0)**2.0<PerfectRad**2.0 or\
        (coordA[i,1]-4e-2)**2.0<PerfectRad**2.0:
        #(coordA[i,0]-0.0)**2.0+(coordA[i,1]-9e-6)**2.0<PerfectRad**2.0 or\
        #(coordA[i,0]-9e-6)**2.0+(coordA[i,1]-9e-6)**2.0<PerfectRad**2.0:
            PFPerfectField[i]=1

        PF_stored.append(0)
        PF_inner.append(0)
        DeformationHistoryX.append(0)
        DeformationHistoryY.append(0)
        DeformationHistoryZ.append(0)
        V_flag.append(0)
        deformation_x_inner.append(0)
        deformation_y_inner.append(0)
        deformation_z_inner.append(0)
        deformation_x_outer.append(0)
        deformation_y_outer.append(0)
        deformation_z_outer.append(0)
        strain_trace.append(0)
        ElasticEnergyField.append(0)
        EnergyHistoryField.append(0)
        
        if PFHistoryField[i]==1.0:
            E_local.append(E)
            nu_local.append(nu)
            K_local.append(\
            #9.0*K*G/(3.0*K+4.0*G)
            K
            )
            Lamda_local.append(E_local[i]*nu_local[i]/(1+nu_local[i])/(1-2.0*nu_local[i]) )
            G_local.append(E_local[i]/(2.*(1+nu_local[i])))
        else :
            E_local.append(E*(PFHistoryField[i]**2.0+StiffnessResidual))
            nu_local.append(nu)
            K_local.append(\
            #9.0*K*G/(3.0*K+4.0*G)
            K*(PFHistoryField[i]**2.0+StiffnessResidual)
            )
            Lamda_local.append(E_local[i]*nu_local[i]/(1+nu_local[i])/(1-2.0*nu_local[i])*(PFHistoryField[i]**2.0+StiffnessResidual) )
            G_local.append(E_local[i]/(2.*(1+nu_local[i]))*(PFHistoryField[i]**2.0+StiffnessResidual))
            etaFieldsA[i]=G_local[i]
            eta1FieldsA[i]=Lamda_local[i]

        for fiber_count in range(0,NumofFiber) :
            if((coordA[i,0]-fiber_x[fiber_count])**2+\
            (coordA[i,1]-fiber_y[fiber_count])**2)<fiber_r[fiber_count]**2:
                E_local[i]=E_fiber
                nu_local[i]=nu_fiber
                fractureToughnessField[i]=cFED*100.0

##########################################################################################
# End of Model Initialization
##########################################################################################
Total_count = 0
Displacement = 0
ExternalStress = 0
for nstep in range(0,numSteps):

   if nstep<OUpLimit:
       Displacement=Displacement+DispStep
       ExternalStress=ExternalStress+ StressStep
   else :
       Displacement=Displacement+DispStep/DispReFactor
       ExternalStress=ExternalStress+ StressStep
   if rank_id==0:
       print "----------Starting step: ",nstep, "Displacement: ",Displacement

   #for id in [beamTop]:
   #    if id in StructurebcMap:
   #        bc = StructurebcMap[id]
   #        bc['specifiedYYTraction'] = LoadCoef*ExternalStress
           #bc['specifiedYDeformation'] = Displacement
   #for id in [beamFront]:
   #    if id in StructurebcMap:
   #        bc = StructurebcMap[id]
   #        bc['specifiedZZTraction'] = ExternalStress
   #for id in [beamRight]:
   #    if id in StructurebcMap:
   #        bc = StructurebcMap[id]
   #        bc['specifiedXXTraction'] = ExternalStress
   #for id in [beamBot]:
   #    if id in StructurebcMap:
   #        bc = StructurebcMap[id]
   #        bc['specifiedYYTraction'] =ExternalStress
   
   for i in range(0,Count):
       #if i == 0:
       #    print "Previous Temperature", temperatureFieldsA[i]
       temperatureFieldsA[i] = temperatureFieldsA[i] + TempStep
       #if i == 0:
       #    print "Current Temperature", temperatureFieldsA[i]       
##########################################################################################
# Start of Middle Loop Iteration
##########################################################################################           
   mid_iter=0
   mid_loop_flag[0] = 1
   repeat_array = [0 for col in range(Count)]
   while (mid_loop_flag[0] ==1) :
       mid_iter=mid_iter+1
       if rank_id==0 :
           print "----------Current step: ",nstep, "Mid loop step: ",mid_iter
##########################################################################################
# Start of Structure Outer Loop Iteration
##########################################################################################
       struct_override_count = 0 
       struct_outer_iter=0
       struct_outer_flag[0]=1
       while (struct_outer_flag[0] ==1) :
           struct_outer_iter=struct_outer_iter+1
           struct_outer_flag[0]=0
           struct_outer_tol_flag[0] = 0
           compress_found_flag[0] = 0
           
           deformFields=structureFields.deformation[cellSitesLocal[n]]
           deformFieldsA=deformFields.asNumPyArray()
           if struct_outer_iter!=1:
               for i in range(0,Count):     
                   deformation_x_outer[i] = deformFieldsA[i][0]
                   deformation_y_outer[i] = deformFieldsA[i][1]
                   deformation_z_outer[i] = deformFieldsA[i][2]
##########################################################################################
# Start of Structure Inner Loop Iteration
##########################################################################################                   
           struct_inner_iter=0
           struct_inner_flag[0] = 1
           while(struct_inner_flag[0]==1):
               struct_inner_iter=struct_inner_iter+1
               if StructIterFlag == 0:
                   numStructIterations = 1

               for niter_struct in range(0,numStructIterations):
                   smodel.advance(1)
               for i in range(0,Count): 
                   deformation_x_inner[i] = deformFieldsA[i][0]
                   deformation_y_inner[i] = deformFieldsA[i][1]
                   deformation_z_inner[i] = deformFieldsA[i][2]
               struct_inner_flag[0] = 0
               smodel.advance(1)
               
               deformation_change_max[0]=0
               deformation_change_maxi=0
               for i in range(0,Count):
                   if fabs((deformation_x_inner[i] - deformFieldsA[i][0])/DeformUnit) > StructTolerance :
                       struct_inner_flag[0] = 1
                   if fabs(deformation_x_inner[i] - deformFieldsA[i][0]) > deformation_change_max[0]:
                       deformation_change_max[0]=fabs(deformation_x_inner[i] - deformFieldsA[i][0])
                       deformation_change_maxi=i
                   if fabs((deformation_y_inner[i] - deformFieldsA[i][1])/DeformUnit) > StructTolerance :
                       struct_inner_flag[0] = 1
                   if fabs(deformation_y_inner[i] - deformFieldsA[i][1]) > deformation_change_max[0]:
                       deformation_change_max[0]=fabs(deformation_y_inner[i] - deformFieldsA[i][1])
                       deformation_change_maxi=i
                   if fabs((deformation_z_inner[i] - deformFieldsA[i][2])/DeformUnit) > StructTolerance :
                       struct_inner_flag[0] = 1
                   if fabs(deformation_z_inner[i] - deformFieldsA[i][2]) > deformation_change_max[0]:
                       deformation_change_max[0]=fabs(deformation_z_inner[i] - deformFieldsA[i][2])
                       deformation_change_maxi=i
               
               MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[deformation_change_max, MPI.DOUBLE], op=MPI.MAX)
               MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[struct_inner_flag, MPI.DOUBLE], op=MPI.MAX)
               if rank_id==0:
                   if struct_inner_flag[0] == 1 :
                       print "Structure inner loop keeps iterating ",deformation_change_max[0],deformation_change_max[0]/DeformUnit,StructTolerance,\
                       coordA[deformation_change_maxi][0],coordA[deformation_change_maxi][1]
                   if struct_inner_flag[0] == 0 :
                       print "Structure inner loop finished ",deformation_change_max[0],deformation_change_max[0]/DeformUnit,StructTolerance,\
                       coordA[deformation_change_maxi][0],coordA[deformation_change_maxi][1],
##########################################################################################
# End of Structure Inner Loop Iteration
##########################################################################################    
           # Traction,Strain
           for n in range(0,nmesh):
               smodel.getStrain(meshes[n])
               smodel.getTraction(meshes[n])

           # Verify Structure Model
           for n in range(0,nmesh):
               tractXFields = structureFields.tractionX[cellSitesLocal[n]]
               tractXFieldsA = tractXFields.asNumPyArray()
               tractYFields = structureFields.tractionY[cellSitesLocal[n]]
               tractYFieldsA = tractYFields.asNumPyArray()
               tractZFields = structureFields.tractionZ[cellSitesLocal[n]]
               tractZFieldsA = tractZFields.asNumPyArray() 
               strainXFields = structureFields.strainX[cellSitesLocal[n]]
               strainXFieldsA = strainXFields.asNumPyArray()  
               strainYFields = structureFields.strainY[cellSitesLocal[n]]
               strainYFieldsA = strainYFields .asNumPyArray() 
               strainZFields = structureFields.strainZ[cellSitesLocal[n]]
               strainZFieldsA = strainZFields.asNumPyArray()
        
               sourceField = fractureFields.source[cellSitesLocal[n]]
               sourceFieldA = sourceField.asNumPyArray()
               sourceCoefField = fractureFields.sourcecoef[cellSitesLocal[n]]
               sourceCoefFieldA = sourceCoefField.asNumPyArray()
               PhaseField = fractureFields.phasefieldvalue[cellSitesLocal[n]]
               PhaseFieldA = PhaseField.asNumPyArray()
           
               deformation_change_max[0]=0
               deformation_change_maxi=0
               #Find out if any cell is in compression
               for i in range(0,Count):
                   strain_trace[i]=strainXFieldsA[i][0]+strainYFieldsA[i][1]+strainZFieldsA[i][2]
                   if abs(deformFieldsA[i,0]-deformation_x_outer[i])>deformation_change_max[0]:
                       deformation_change_max[0]=abs(deformFieldsA[i,0]-deformation_x_outer[i])
                       deformation_change_maxi=i
                   if abs(deformFieldsA[i,1]-deformation_y_outer[i])>deformation_change_max[0]:
                       deformation_change_max[0]=abs(deformFieldsA[i,1]-deformation_y_outer[i]) 
                       deformation_change_maxi=i                 
                   if abs(deformFieldsA[i,2]-deformation_z_outer[i])>deformation_change_max[0]:
                       deformation_change_max[0]=abs(deformFieldsA[i,2]-deformation_z_outer[i]) 
                       deformation_change_maxi=i            
               
               #print "rank: ",rank_id, "Max Deformation: ", deformation_change_max[0]/DeformUnit,deformation_change_max[0],\
               #"Current Deformation Component: ",deformFieldsA[deformation_change_maxi,0],deformFieldsA[deformation_change_maxi,1],deformFieldsA[deformation_change_maxi,2],\
               #"Previous Deformation Component",deformation_x_loop[deformation_change_maxi],deformation_y_loop[deformation_change_maxi],deformation_z_loop[deformation_change_maxi],\
               #"Phase Field: ", PhaseFieldA[deformation_change_maxi],strain_trace[deformation_change_maxi]
               if deformation_change_max[0]/DeformUnit<StructOuterTolerance:
                   struct_outer_tol_flag[0]=1

               MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[deformation_change_max, MPI.DOUBLE], op=MPI.MAX)
               MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[compress_found_flag, MPI.DOUBLE], op=MPI.MAX)
               MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[struct_outer_tol_flag, MPI.DOUBLE], op=MPI.MIN)
           
               if struct_outer_iter==1:
                   struct_outer_flag[0]=1
                   if rank_id==0:
                       print "Doing it one more time"           
               elif compress_found_flag[0]==1 and struct_outer_tol_flag[0] == 0:
                   struct_outer_flag[0]=1
                   if rank_id==0:
                       print "Skipping fracture model from compress-found",deformation_change_max[0]/DeformUnit,deformation_change_max[0]
               elif compress_found_flag[0]==0 and struct_outer_tol_flag[0] == 0 and struct_override_count<StructIterUpLimit:
                   struct_outer_flag[0]=1
                   struct_override_count=struct_override_count+1
                   if rank_id==0:
                       print "Skipping fracture model from tolerance",deformation_change_max[0]/DeformUnit,deformation_change_max[0],struct_override_count
               elif compress_found_flag[0]==0 and struct_outer_tol_flag[0] == 0 and struct_override_count>=StructIterUpLimit:
                   if rank_id==0:
                       print "Getting out of structure model But violating tolerance",deformation_change_max[0]/DeformUnit,deformation_change_max[0],struct_override_count
               elif compress_found_flag[0]==1 and struct_outer_tol_flag[0] == 1:
                   if rank_id==0:
                       print "Getting out of structure model But violating compress-found ",deformation_change_max[0]/DeformUnit,deformation_change_max[0]
               else:
                   if rank_id==0:
                       print "Getting out of structure model ",deformation_change_max[0]/DeformUnit,deformation_change_max[0],\
                       coordA[deformation_change_maxi][0],coordA[deformation_change_maxi][1],PhaseFieldA[deformation_change_maxi]
##########################################################################################
# End of Structure Outer Loop Iteration
##########################################################################################                
       #Gather loading and elastic energy info
       LoadingTop  = array([0.0])
       LoadingRight  = array([0.0])
       LoadingBottom  = array([0.0])
       LoadingLeft  = array([0.0])
       Loading_count_top = array([0.0])
       Loading_count_right = array([0.0])
       Loading_count_bottom = array([0.0])
       Loading_count_left = array([0.0])
       
       DispTop = array([0.0])
       DispRight = array([0.0])
       DispBottom = array([0.0])
       DispLeft = array([0.0])
       Disp_count_top = array([0.0])
       Disp_count_right = array([0.0])
       Disp_count_bottom = array([0.0])
       Disp_count_left = array([0.0]) 
       
       Total_Elastic_Energy = array([0.0])
       Total_Compression_Elastic_Energy = array([0.0])
       Max_Vol_Stress = array([-1e20])
       Max_Vol_Stress_X = array([0.0])
       Max_Vol_Stress_Y = array([0.0])
       Max_Dev_Stress = array([0.0])
       Max_Dev_Stress_X = array([0.0])
       Max_Dev_Stress_Y = array([0.0])
       Max_Vol_Strain = array([-100.0])
       Max_Vol_Strain_X = array([0.0])
       Max_Vol_Strain_Y = array([0.0])
       Max_Dev_Strain = array([0.0])
       Max_Dev_Strain_X = array([0.0])
       Max_Dev_Strain_Y = array([0.0])
        
       
       for i in range(0,Count):

           if strain_trace[i] > 0:
               strain_trace_positive=strain_trace[i]
               strain_trace_negative=0
               #V_flag[i] = 0
           else :
               strain_trace_positive=0
               strain_trace_negative=strain_trace[i]  
               #V_flag[i] = 1               
           strain_trace_mean=strain_trace[i]/3.0
           strain_dev2_trace=(strainXFieldsA[i][0]-strain_trace_mean)**2+strainXFieldsA[i][1]**2+strainXFieldsA[i][2]**2+\
           strainYFieldsA[i][0]**2+(strainYFieldsA[i][1]-strain_trace_mean)**2+strainYFieldsA[i][2]**2+\
           strainZFieldsA[i][0]**2+strainZFieldsA[i][1]**2+(strainZFieldsA[i][2]-strain_trace_mean)**2
           strain_2_trace=(strainXFieldsA[i][0])**2+strainXFieldsA[i][1]**2+strainXFieldsA[i][2]**2+\
           strainYFieldsA[i][0]**2+(strainYFieldsA[i][1])**2+strainYFieldsA[i][2]**2+\
           strainZFieldsA[i][0]**2+strainZFieldsA[i][1]**2+(strainZFieldsA[i][2])**2

           if SymFlag==1:
               if strain_trace[i] >0:
                   ElasticEnergyField[i] = K_local[i]/2.0*strain_trace_positive**2+G_local[i]*strain_dev2_trace
               else: 
                   ElasticEnergyField[i] = K_local[i]/2.0*strain_trace_negative**2+G_local[i]*strain_dev2_trace
               if i==100:
                   print i,ElasticEnergyField[i]
                   print strainXFieldsA[i],strainYFieldsA[i],strainZFieldsA[i]
                   print tractZFieldsA[i][2]
                  # print eigenvector1FieldsA[i],eigenvector2FieldsA[i],eigenvector3FieldsA[i]
                   #print tractXFieldsA[i][0],tractYFieldsA[i][1],tractZFieldsA[i][2],pfvFieldsA[i],V_flag[i]
               Total_Elastic_Energy[0] = Total_Elastic_Energy[0] + (PhaseFieldA[i]**2.0+StiffnessResidual)*(K_local[i]/2.0*strain_trace[i]**2+G_local[i]*strain_dev2_trace)*volumeA[i]
           #else:
           #    if strain_trace[i] >0:
           #        ElasticEnergyField[i] = Lamda_local[i]/2.0*strain_trace_positive**2+G_local[i]*(eigenvalue1_positive[0]**2.0+eigenvalue2_positive[0]**2.0+eigenvalue3_positive[0]**2.0)
           #    else: 
           #        ElasticEnergyField[i] = G_local[i]*(eigenvalue1_positive[0]**2.0+eigenvalue2_positive[0]**2.0+eigenvalue3_positive[0]**2.0)
           #    if i==100:
           #        print i,ElasticEnergyField[i],eigenvalueFieldsA[i]
           #        print strainXFieldsA[i],strainYFieldsA[i],strainZFieldsA[i]
           #        print tractZFieldsA[i][2]
           #        print eigenvector1FieldsA[i],eigenvector2FieldsA[i],eigenvector3FieldsA[i]
           #        #print tractXFieldsA[i][0],tractYFieldsA[i][1],tractZFieldsA[i][2],pfvFieldsA[i],V_flag[i]
           #    Total_Elastic_Energy[0] = Total_Elastic_Energy[0] + ((PhaseFieldA[i]**2.0+StiffnessResidual)*(K_local[i]/2.0*strain_trace_positive**2+G_local[i]*strain_dev2_trace)+K_local[i]/2.0*strain_trace_negative**2)*volumeA[i]
           
           Total_Compression_Elastic_Energy[0] = Total_Compression_Elastic_Energy[0] + K_local[i]/2.0*strain_trace_negative**2
           
           if coordA[i,0]>0.0*BoundaryPositionRight and coordA[i,0]<1.0*BoundaryPositionRight\
           and coordA[i,1]>0.0*BoundaryPositionTop and coordA[i,1]<1.0*BoundaryPositionTop and i < selfCount:
               if (tractXFieldsA[i][0]+tractYFieldsA[i][1]+tractZFieldsA[i][2])/3.0>Max_Vol_Stress[0]:
                   Max_Vol_Stress[0]=(tractXFieldsA[i][0]+tractYFieldsA[i][1]+tractZFieldsA[i][2])/3.0
                   Max_Vol_Stress_X[0] = coordA[i,0]
                   Max_Vol_Stress_Y[0] = coordA[i,1]
                   Local_MVStress=Max_Vol_Stress[0]
               if (0.5*((tractXFieldsA[i][0]-tractYFieldsA[i][1])**2.0+(tractYFieldsA[i][1]-tractZFieldsA[i][2])**2.0+(tractZFieldsA[i][2]-tractXFieldsA[i][0])**2.0)+\
               3.0*(tractXFieldsA[i][1]**2.0+tractXFieldsA[i][2]**2.0+tractYFieldsA[i][2]**2.0))**0.5>Max_Dev_Stress[0]:
                   Max_Dev_Stress[0]=(0.5*((tractXFieldsA[i][0]-tractYFieldsA[i][1])**2.0+(tractYFieldsA[i][1]-tractZFieldsA[i][2])**2.0+(tractZFieldsA[i][2]-tractXFieldsA[i][0])**2.0)+\
                   3.0*(tractXFieldsA[i][1]**2.0+tractXFieldsA[i][2]**2.0+tractYFieldsA[i][2]**2.0))**0.5
                   Max_Dev_Stress_X[0] = coordA[i,0]
                   Max_Dev_Stress_Y[0] = coordA[i,1]
                   Local_MDStress=Max_Dev_Stress[0]
               if (strainXFieldsA[i][0]+strainYFieldsA[i][1]+strainZFieldsA[i][2])>Max_Vol_Strain[0]:
                   Max_Vol_Strain[0]=(strainXFieldsA[i][0]+strainYFieldsA[i][1]+strainZFieldsA[i][2])
                   Max_Vol_Strain_X[0] = coordA[i,0]
                   Max_Vol_Strain_Y[0] = coordA[i,1]
                   Local_MVStrain=Max_Vol_Strain[0]
               if (0.5*((strainXFieldsA[i][0]-strainYFieldsA[i][1])**2.0+(strainYFieldsA[i][1]-strainZFieldsA[i][2])**2.0+(strainZFieldsA[i][2]-strainXFieldsA[i][0])**2.0)+\
               3.0*(strainXFieldsA[i][1]**2.0+strainXFieldsA[i][2]**2.0+strainYFieldsA[i][2]**2.0))**0.5>Max_Dev_Strain[0]:
                   Max_Dev_Strain[0]=(0.5*((strainXFieldsA[i][0]-strainYFieldsA[i][1])**2.0+(strainYFieldsA[i][1]-strainZFieldsA[i][2])**2.0+(strainZFieldsA[i][2]-strainXFieldsA[i][0])**2.0)+\
                   3.0*(strainXFieldsA[i][1]**2.0+strainXFieldsA[i][2]**2.0+strainYFieldsA[i][2]**2.0))**0.5
                   Max_Dev_Strain_X[0] = coordA[i,0]
                   Max_Dev_Strain_Y[0] = coordA[i,1]
                   Local_MDStrain=Max_Dev_Strain[0]        
           if coordA[i,1]> BoundaryPositionTop and i < selfCount :
               LoadingTop[0] += tractYFieldsA[i][1]
               Loading_count_top[0]=Loading_count_top[0]+1
               DispTop[0] += deformFieldsA[i][1]
               Disp_count_top[0] = Disp_count_top[0]+1
           elif coordA[i,0]> BoundaryPositionRight and i < selfCount :
               LoadingRight[0] += tractXFieldsA[i][0]
               Loading_count_right[0]=Loading_count_right[0]+1
               DispRight[0] += deformFieldsA[i][0]
               Disp_count_right[0] = Disp_count_right[0]+1
           elif coordA[i,0]< BoundaryPositionLeft and i < selfCount :
               LoadingLeft[0] += tractXFieldsA[i][0]
               Loading_count_left[0]=Loading_count_left[0]+1
               DispLeft[0] += deformFieldsA[i][0]
               Disp_count_left[0] = Disp_count_left[0]+1
           elif coordA[i,1]< BoundaryPositionBottom and i < selfCount :
               LoadingBottom[0] += tractYFieldsA[i][1]
               Loading_count_bottom[0]=Loading_count_bottom[0]+1
               DispBottom[0] += deformFieldsA[i][1]
               Disp_count_bottom[0] = Disp_count_bottom[0]+1
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[Loading_count_top, MPI.DOUBLE], op=MPI.SUM)
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[LoadingTop, MPI.DOUBLE], op=MPI.SUM)
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[Loading_count_right, MPI.DOUBLE], op=MPI.SUM)
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[LoadingRight, MPI.DOUBLE], op=MPI.SUM)
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[Loading_count_bottom, MPI.DOUBLE], op=MPI.SUM)
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[LoadingBottom, MPI.DOUBLE], op=MPI.SUM)
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[Loading_count_left, MPI.DOUBLE], op=MPI.SUM)
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[LoadingLeft, MPI.DOUBLE], op=MPI.SUM)
       
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[Disp_count_top, MPI.DOUBLE], op=MPI.SUM)
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[DispTop, MPI.DOUBLE], op=MPI.SUM)
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[Disp_count_right, MPI.DOUBLE], op=MPI.SUM)
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[DispRight, MPI.DOUBLE], op=MPI.SUM)
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[Disp_count_bottom, MPI.DOUBLE], op=MPI.SUM)
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[DispBottom, MPI.DOUBLE], op=MPI.SUM)
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[Disp_count_left, MPI.DOUBLE], op=MPI.SUM)
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[DispLeft, MPI.DOUBLE], op=MPI.SUM)

       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[Total_Elastic_Energy, MPI.DOUBLE], op=MPI.SUM)
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[Total_Compression_Elastic_Energy, MPI.DOUBLE], op=MPI.SUM)
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[Max_Vol_Stress, MPI.DOUBLE], op=MPI.MAX) 
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[Max_Dev_Stress, MPI.DOUBLE], op=MPI.MAX) 
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[Max_Vol_Strain, MPI.DOUBLE], op=MPI.MAX) 
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[Max_Dev_Strain, MPI.DOUBLE], op=MPI.MAX) 
       
       if Local_MVStress!=Max_Vol_Stress[0] :
           Max_Vol_Stress_X[0] = -1e20
           Max_Vol_Stress_Y[0] = -1e20
       if Local_MDStress!=Max_Dev_Stress[0] :
           Max_Dev_Stress_X[0] = -1e20
           Max_Dev_Stress_Y[0] = -1e20
       if Local_MVStrain!=Max_Vol_Strain[0] :
           Max_Vol_Strain_X[0] = -1e20
           Max_Vol_Strain_Y[0] = -1e20 
       if Local_MDStrain!=Max_Dev_Strain[0] :
           Max_Dev_Strain_X[0] = -1e20
           Max_Dev_Strain_Y[0] = -1e20    
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[Max_Vol_Stress_X, MPI.DOUBLE], op=MPI.MAX) 
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[Max_Vol_Stress_Y, MPI.DOUBLE], op=MPI.MAX) 
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[Max_Dev_Stress_X, MPI.DOUBLE], op=MPI.MAX) 
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[Max_Dev_Stress_Y, MPI.DOUBLE], op=MPI.MAX)
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[Max_Vol_Strain_X, MPI.DOUBLE], op=MPI.MAX) 
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[Max_Vol_Strain_Y, MPI.DOUBLE], op=MPI.MAX) 
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[Max_Dev_Strain_X, MPI.DOUBLE], op=MPI.MAX) 
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[Max_Dev_Strain_Y, MPI.DOUBLE], op=MPI.MAX)     
       LoadingTop[0]=LoadingTop[0]/Loading_count_top[0]
       LoadingRight[0]=LoadingRight[0]/Loading_count_right[0]
       LoadingBottom[0]=LoadingBottom[0]/Loading_count_bottom[0]
       LoadingLeft[0]=LoadingLeft[0]/Loading_count_left[0]
       DispTop[0]=DispTop[0]/Disp_count_top[0]
       DispRight[0]=DispRight[0]/Disp_count_right[0]
       DispBottom[0]=DispBottom[0]/Disp_count_bottom[0]
       DispLeft[0]=DispLeft[0]/Disp_count_left[0]
       #End of gathering loading and elastic energy info
       if rank_id == 0:
           print  "Loading Force: ",LoadingTop[0],LoadingRight[0],LoadingLeft[0],LoadingBottom[0]
           print  "Displacement: ",DispTop[0],DispRight[0],DispLeft[0],DispBottom[0]
   
       #Update SourceCoef for Fracture Model   
       for i in range(0,Count):  
           sourceCoefFieldA[i]=-(4.0*cLoC*ElasticEnergyField[i]/cFED+1.0)                 
 ########################################################################################## 
# Start of the fracture model
 ########################################################################################## 
       for i in range(0,Count): 
           PF_stored[i]=PhaseFieldA[i]

       fract_inner_flag[0] = 1
       PF_change_max_inner[0] = 0
       while fract_inner_flag[0] == 1:
           for niter_PF in range(0,numPFIterations):
               for n in range(0,nmesh):
                   for i in range(0,Count):
                       if PhaseFieldA[i]<PFPerfectField[i]:
                           PhaseFieldA[i]=PFPerfectField[i]
                       if PhaseFieldA[i]>PFHistoryField[i]:
                           PhaseFieldA[i]=PFHistoryField[i]
                       sourceFieldA[i]=-(4.0*cLoC*ElasticEnergyField[i]/cFED+1.0)*PhaseFieldA[i]
               tmodel.advance(1)
               
           for i in range(0,Count):
               if PhaseFieldA[i]<PFPerfectField[i]:
                   PhaseFieldA[i]=PFPerfectField[i]
               if PhaseFieldA[i]>PFHistoryField[i]:
                   PhaseFieldA[i]=PFHistoryField[i]
               PF_inner[i]=PhaseFieldA[i]
                   
           for n in range(0,nmesh):
               for i in range(0,Count):
                   sourceFieldA[i]=-(4.0*cLoC*ElasticEnergyField[i]/cFED+1.0)*PhaseFieldA[i]
           tmodel.advance(1)
           
           fract_inner_flag[0] = 0
           for i in range(0,Count):
               if PhaseFieldA[i]<PFPerfectField[i]:
                   PhaseFieldA[i]=PFPerfectField[i]
               if PhaseFieldA[i]>PFHistoryField[i]:
                   PhaseFieldA[i]=PFHistoryField[i]
               if abs(PF_inner[i]-PhaseFieldA[i])>PFTolerance:
                   fract_inner_flag[0] = 1
               if PF_change_max_inner[0]< abs(PF_inner[i]-PhaseFieldA[i]):
                   PF_change_max_inner[0] = abs(PF_inner[i]-PhaseFieldA[i])
           MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[fract_inner_flag, MPI.DOUBLE], op=MPI.MAX)
           MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[PF_change_max_inner, MPI.DOUBLE], op=MPI.MAX)
           if rank_id==0:
               if fract_inner_flag[0] == 1:
                   print "Fracture model keeps iterating ",PF_change_max_inner[0]
               else :
                   print "Fracture model ends ",PF_change_max_inner[0]
 ########################################################################################## 
 #End of Phase Field Inner Loop
 ##########################################################################################  
       PF_min[0] = 1.0
       PF_change_max[0]=0
       PF_change_maxi=0
       for i in range(0,Count):
           if PhaseFieldA[i]<PF_min[0] and i<selfCount:
               PF_min[0]=PhaseFieldA[i]
           if abs(PhaseFieldA[i]-PF_stored[i]) > PF_change_max[0] :
               PF_change_max[0]=abs(PhaseFieldA[i]-PF_stored[i])
               PF_change_maxi=i
       
       if PF_change_max[0] < PFOuterTolerance or PFIterFlag == 0 :
       #if mid_iter==1:
           mid_loop_flag[0] = 0
       
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[PF_min, MPI.DOUBLE], op=MPI.MIN) 
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[mid_loop_flag, MPI.DOUBLE], op=MPI.MAX)
       MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[PF_change_max, MPI.DOUBLE], op=MPI.MAX)
       
       if rank_id==0:
           print "Phase Field Minimum Value: ",PF_min[0], "Maximum Phase Field Change: ",PF_change_max[0],PhaseFieldA[PF_change_maxi],PhaseFieldA[i]
       if mid_iter>MidIterUpLimit:
           mid_loop_flag[0] = 0
    
       #Update Modified Elastic Modulus in Structure Module            
       for i in range(0,Count):
           #etaFieldsA[i]=G_local[i]*(PhaseFieldA[i]**2.0+StiffnessResidual)
           #eta1FieldsA[i]=Lamda_local[i]*PhaseFieldA[i]**2.0
           if SymFlag==1:
               #etaFieldsA[i]=G_local[i]
               etaFieldsA[i]=G_local[i]*(PhaseFieldA[i]**2.0+StiffnessResidual)
               eta1FieldsA[i]=Lamda_local[i]*(PhaseFieldA[i]**2.0+StiffnessResidual)
           #elif V_flag[i]==0:
           #    etaFieldsA[i]=G_local[i]
           #    eta1FieldsA[i]=Lamda_local[i]
           #    #etaFieldsA[i]=G_local[i]*(PhaseFieldA[i]**2.0+StiffnessResidual)
           #    #eta1FieldsA[i]=Lamda_local[i]*(PhaseFieldA[i]**2.0+StiffnessResidual)
           else :
               etaFieldsA[i]=G_local[i]
               eta1FieldsA[i]=Lamda_local[i]
               #etaFieldsA[i]=G_local[i]*(PhaseFieldA[i]**2.0+StiffnessResidual)
               #eta1FieldsA[i]=Lamda_local[i]*(PhaseFieldA[i]**2.0+StiffnessResidual)
               #eta1FieldsA[i]=Lamda_local[i]+G_local[i]*2.0/3.0*(1-(PhaseFieldA[i]**2.0+StiffnessResidual))
           #pfvFieldsA[i]=PhaseFieldA[i]
##########################################################################################  
       #output time change and intermediate status
       title_name="Inter "+str(nstep)+" "+str(mid_iter)
       if PF_change_max[0]>OPFLimit or nstep>=OUpLimit:
           MidOInterval=MidOInterval_s
       else :
           MidOInterval=MidOInterval_l
       #Output Structure Module 
       if mid_iter % MidOInterval ==0:
           tecplotEntireStructureDomainPara.dumpTecplotEntireStructureDomain(nmesh,  meshes, fluent_meshes, options.type, structureFields,structure_file_name,title_name,Total_count) 
       #Output Fracture Module 
       if mid_iter % MidOInterval ==0:
           tecplotEntireFractureDomainPara.dumpTecplotEntireFractureDomain(nmesh,  meshes, fluent_meshes, options.type, fractureFields,fracture_file_name,title_name,Total_count)
       #Output equilibrium status VTK files
       if mid_iter % MidOInterval ==0:   
           writer_fracture_iter = exporters.VTKWriterA(geomFields,meshes,"fracture-inter-"+str(nstep)+"-"+str(mid_iter)+"-"+".vtk","fracture output",False,0)
           writer_fracture_iter.init()
           writer_fracture_iter.writeScalarField(fractureFields.phasefieldvalue,"PhaseField")
           writer_fracture_iter.writeVectorField(structureFields.deformation,"Deformation")
           writer_fracture_iter.finish()
           writer_structure_iter = exporters.VTKWriterA(geomFields,meshes,"structure-inter-"+str(nstep)+"-"+str(mid_iter)+"-"+".vtk","structure output",False,0)
           writer_structure_iter.init()
           writer_structure_iter.writeVectorField(structureFields.deformation,"Deformation")
           writer_structure_iter.writeVectorField(structureFields.strainX,"StrainX")
           writer_structure_iter.writeVectorField(structureFields.strainY,"StrainY")
           writer_structure_iter.writeVectorField(structureFields.strainZ,"StrainZ")
           writer_structure_iter.writeVectorField(structureFields.tractionX,"TractionX")
           writer_structure_iter.writeVectorField(structureFields.tractionY,"TractionY")
           writer_structure_iter.writeVectorField(structureFields.tractionZ,"TractionZ")       
           writer_structure_iter.finish() 
           Total_count = Total_count +1
       if rank_id == 0 :
           t1 = time.time()
           print "TIME ELAPSE: ",t1-t0
           #write intermediate status file
           ss_structure.write(str(Displacement) + " " + str(LoadingTop[0]) + " "+ str(LoadingRight[0]) + " " +str(LoadingLeft[0])+ " " +str(LoadingBottom[0])+ " "  + str(t1-t0) + " " +str(Max_Vol_Stress[0])+ " "+str(Max_Dev_Stress[0])+ " "+ str(PF_min[0]) + " "+ str(PF_change_max[0]) +"\n")
           ss_structure.flush()  
       #End of Output Intermediate Status
##########################################################################################
#End of Middle Loop
##########################################################################################
   #Output Equilibrium Status
   PF_equil_change_max=array([0.0])
   deformation_equil_change_max=array([0.0])
   #Update energy and phase field history field
   for i in range(0,Count):
       #EnergyHistoryField[i]=ElasticEnergyField[i]
       if abs(PFHistoryField[i]-PhaseFieldA[i]) >PF_equil_change_max[0]:
           PF_equil_change_max[0]=abs(PFHistoryField[i]-PhaseFieldA[i])
       PFHistoryField[i]=PhaseFieldA[i]

       if abs(deformFieldsA[i,0]-DeformationHistoryX[i])>deformation_equil_change_max[0]:
           deformation_equil_change_max[0]=abs(deformFieldsA[i,0]-DeformationHistoryX[i])
       if abs(deformFieldsA[i,1]-DeformationHistoryY[i])>deformation_equil_change_max[0]:
           deformation_equil_change_max[0]=abs(deformFieldsA[i,1]-DeformationHistoryY[i])                  
       if abs(deformFieldsA[i,2]-DeformationHistoryZ[i])>deformation_equil_change_max[0]:
           deformation_equil_change_max[0]=abs(deformFieldsA[i,2]-DeformationHistoryZ[i])   
       DeformationHistoryX[i]=deformFieldsA[i][0]
       DeformationHistoryY[i]=deformFieldsA[i][1]
       DeformationHistoryZ[i]=deformFieldsA[i][2]
   MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[deformation_equil_change_max, MPI.DOUBLE], op=MPI.MAX)
   MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,[PF_equil_change_max, MPI.DOUBLE], op=MPI.MAX)
   
   if PF_equil_change_max[0]>OPFLimit:
       OInterval=OInterval_s
   else :
       if nstep>=OUpLimit:
           OInterval=OInterval_s
       else :
           OInterval=OInterval_l
           
   if rank_id == 0 :
       #write stress-strain curve
       sp_structure.write(str(nstep) + " "+ str(PF_min[0]) + " "+ str(Total_Elastic_Energy[0]) +" "+\
       str(LoadingTop[0]) + " "+ str(LoadingRight[0]) + " " +str(LoadingLeft[0])+ " " +str(LoadingBottom[0])+" " +\
       str(DispTop[0]) + " "+ str(DispRight[0]) + " " +str(DispLeft[0])+ " " +str(DispBottom[0])+" " +\
       str(Max_Vol_Stress[0])+ " " +str(Max_Vol_Stress_X[0])+ " " +str(Max_Vol_Stress_Y[0])+ " " +\
       str(Max_Dev_Stress[0])+ " " +str(Max_Dev_Stress_X[0])+ " " +str(Max_Dev_Stress_Y[0])+ " " +\
       str(Max_Vol_Strain[0])+ " " +str(Max_Vol_Strain_X[0])+ " " +str(Max_Vol_Strain_Y[0])+ " " +\
       str(Max_Dev_Strain[0])+ " " +str(Max_Dev_Strain_X[0])+ " " +str(Max_Dev_Strain_Y[0])+ " " +\
       "\n")
       sp_structure.flush()         
   title_name="Equil "+str(nstep)
   #Output Structure Module 
   if nstep % OInterval ==0:
       tecplotEntireStructureDomainPara.dumpTecplotEntireStructureDomain(nmesh,  meshes, fluent_meshes, options.type, structureFields,structure_file_name,title_name,Total_count) 
   #Output Fracture Module 
   if nstep % OInterval ==0:
       tecplotEntireFractureDomainPara.dumpTecplotEntireFractureDomain(nmesh,  meshes, fluent_meshes, options.type, fractureFields,fracture_file_name,title_name,Total_count)
   #Output equilibrium status VTK files
   if nstep % OInterval ==0:   
       writer_fracture = exporters.VTKWriterA(geomFields,meshes,"fracture-equil-"+str(nstep)+".vtk","fracture output",False,0)
       writer_fracture.init()
       writer_fracture.writeScalarField(fractureFields.phasefieldvalue,"PhaseField")
       writer_fracture.writeVectorField(structureFields.deformation,"Deformation")
       writer_fracture.finish()
       writer_structure = exporters.VTKWriterA(geomFields,meshes,"structure-equil-"+str(nstep)+".vtk","structure output",False,0)
       writer_structure.init()
       writer_structure.writeVectorField(structureFields.deformation,"Deformation")
       writer_structure.writeVectorField(structureFields.strainX,"StrainX")
       writer_structure.writeVectorField(structureFields.strainY,"StrainY")
       writer_structure.writeVectorField(structureFields.strainZ,"StrainZ")
       writer_structure.writeVectorField(structureFields.tractionX,"TractionX")
       writer_structure.writeVectorField(structureFields.tractionY,"TractionY")
       writer_structure.writeVectorField(structureFields.tractionZ,"TractionZ")      
       writer_structure.finish()  
       Total_count = Total_count +1
   #End of Output Equilibrium Status   
