import numpy as np
import vtk
import time
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindowInteractor, vtkRenderWindow, vtkPolyDataMapper, vtkActor
from vtkmodules.vtkIOPLY import vtkPLYReader
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkIterativeClosestPointTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter

def GetPD(filename):
    plyReader = vtkPLYReader()
    plyReader.SetFileName(filename)
    plyReader.Update()
    return plyReader.GetOutput()

def GetActor(inputPD, R, G, B, scale):
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(inputPD)

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(R, G, B)
    actor.GetProperty().SetPointSize(scale)

    return actor

##################################################################
##                    Testing interface                         ##

bunnyOG_PD = GetPD("LV5\\models\\bunny.ply")
bunny1_PD = GetPD("LV5\\models\\bunny_t5_parc.ply")

totalIterations = 20
pointPairs = 100

##################################################################


icp = vtkIterativeClosestPointTransform()
icp.SetSource(bunny1_PD) #Ulazni objekt (početna poza objekta)
icp.SetTarget(bunnyOG_PD) #Konačni objekt (željena poza objekta)
icp.GetLandmarkTransform().SetModeToRigidBody() #Potrebni način rada je transformacija za kruta tijela
icp.SetMaximumNumberOfIterations(totalIterations) #Željeni broj iteracija
icp.SetMaximumNumberOfLandmarks(pointPairs) #Koliko parova točaka da se koristi prilikom minimiziranja cost funkcije

totalTime = -time.time()
icp.Update() #Provedi algoritam
totalTime += time.time()

icpTransformFilter = vtkTransformPolyDataFilter()
icpTransformFilter.SetInputData(bunny1_PD) #Objekt s početnim koordinatama
icpTransformFilter.SetTransform(icp) #transformiramo na novi položaj koristeći transformacijsku matricu
icpTransformFilter.Update()

icpResultPD = icpTransformFilter.GetOutput() #Transformirani (novi) objekt

bunnyOG = GetActor(bunnyOG_PD, 0, 0, 1, 5)
# bunny1 = GetActor(bunny1_PD, 1, 0, 0, 5)
bunny1_corr = GetActor(icpResultPD, 0, 1, 0, 5)

renderer = vtkRenderer()
renderer.SetBackground(0.5, 0.5, 0.5)
renderer.AddActor(bunnyOG)
# renderer.AddActor(bunny1)
renderer.AddActor(bunny1_corr)
renderer.ResetCamera() #Centriraj kameru tako da obuhvaća objekte na sceni

window = vtkRenderWindow()
window.AddRenderer(renderer) #Moguće je dodati i više renderera na jedan prozor
window.SetSize(800, 600) #Veličina prozora na ekranu
window.SetWindowName("Scena") #Naziv prozora
window.Render() #Renderaj scenu

interactor = vtkRenderWindowInteractor()
interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
interactor.SetRenderWindow(window)

print(f"Total time: {totalTime} seconds")
interactor.Start() #Pokretanje interaktora, potrebno kako se vtk prozor ne bi odmah zatvorio


