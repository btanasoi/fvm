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

import vtkEntireThermalDomain2DPara

##########################################################################################
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
thermalFields =  models.ThermalFields('therm')
tmodel = models.ThermalModelA(geomFields,thermalFields,meshes)
#structureFields =  models.StructureFields('structure')
#smodel = models.StructureModelA(geomFields, structureFields, meshes)
##########################################################################################
##########################################################################################

timeStep = 1.0                # Size of timestep (seconds)
timeStepN1 = timeStep
timeStepN2 = timeStep
numTimeSteps = 1000               # Number of timesteps in global combined solution

#SQUARE10000
topID = 2
botID = 6
othersID = (4, 5)

bcMap = tmodel.getBCMap()

if 2 in bcMap:
   bc2 = tmodel.getBCMap()[2]
   bc2.bcType = 'SpecifiedTemperature'
   bc2.setVar('specifiedTemperature',110)
if 4 in bcMap:
   bc4 = tmodel.getBCMap()[4]
   #bc4.bcType = 'Symmetry'
   bc4.bcType = 'SpecifiedTemperature'
   bc4.setVar('specifiedTemperature',120)
  
if 5 in bcMap:
   bc5 = tmodel.getBCMap()[5]
   #bc5.bcType = 'Symmetry'
   bc5.bcType = 'SpecifiedTemperature'
   bc5.setVar('specifiedTemperature',100)
 
if 6 in bcMap:
   bc6 = tmodel.getBCMap()[6]
   bc6.bcType = 'SpecifiedTemperature'
   bc6.setVar('specifiedTemperature',100)
   
vcMap = tmodel.getVCMap()
for i,vc in vcMap.iteritems():
    vc = vcMap[i]
    vc.setVar('thermalConductivity',143.0)
    vc.setVar('density',2800.0)
    vc.setVar('specificHeat',795.0)

tSolver = fvmbaseExt.AMG()
#tSolver.smootherType = fvmbaseExt.AMG.JACOBI
tSolver.relativeTolerance = 1e-9
tSolver.nMaxIterations = 200000
tSolver.maxCoarseLevels=20
tSolver.verbosity=0
#tSolver.setMergeLevelSize(40000)

toptions = tmodel.getOptions()
toptions.linearSolver = tSolver
toptions.transient = True
toptions.timeStepN1 = timeStepN1
toptions.timeStepN2 = timeStepN2
toptions.timeDiscretizationOrder = 2
toptions.setVar("timeStep", timeStep)
toptions.setVar("initialTemperature", 100.0)
#toptions.transient = False

tmodel.init()

cellSitesLocal = []

for n in range(0,nmesh):
    cellSitesLocal.append(meshes[n].getCells() )
    Count = cellSitesLocal[n].getCount()
    selfCount = cellSitesLocal[n].getSelfCount()
    coord=geomFields.coordinate[cellSitesLocal[n]]
    coordA=coord.asNumPyArray()
    volume=geomFields.volume[cellSitesLocal[n]]
    volumeA=volume.asNumPyArray()
    source = thermalFields.source[cellSitesLocal[n]]
    sourceA = source.asNumPyArray()
    temperature = thermalFields.temperature[cellSitesLocal[n]]
    temperatureA = temperature.asNumPyArray()
    
for c in range(0, Count):
    if (coordA[c,0]>0.5-0.05 and coordA[c,0]<0.5+0.05) and\
    (coordA[c,1]>0.5-0.05 and coordA[c,1]<0.5+0.05):
        sourceA[c] = 1e6
        #sourceA[c] = 0

for nstep in range(0,numTimeSteps):
    print "step: ",nstep
    tmodel.advance(10)
    if nstep % 10 == 0:
        vtkEntireThermalDomain2DPara.dumpvtkEntireThermalDomain2D(geomFields, nmesh,  meshes, fluent_meshes, options.type, thermalFields, nstep)
    tmodel.updateTime()

# Set Parameters
NX = 100
NY = 100
NZ = 1
thermal_file_name = "tecplotresult_thermal" + ".dat"
f_thermal = open(thermal_file_name, 'w')
f_thermal.write("variables = \"x\", \"y\", \"z\", \"Source\", \"Temperature\" \n") 

#Output Fracture Module
f_thermal.write("ZONE "+"I = "+ str(NX) + "J = "+ str(NY) +"K = "+ str(NZ) +" \n")
for i in range(0,selfCount):
   f_thermal.write(str(coordA[i,0])+" "+\
   str(coordA[i,1])+" "+\
   str(coordA[i,2])+" "+\
   str(sourceA[i])+" "+\
   str(temperatureA[i])+" \n")

#volume = geomFields.volume[cells]
#volumeA = volume.asNumPyArray()
#for c in range(0, Count):
#    print c, volumeA[c]

#pdb.set_trace()
