import numpy as np
import time
import vtk
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

totalTime = 0
dt = 0.05

icp = vtkIterativeClosestPointTransform()

renderer = vtkRenderer()
renderer.SetBackground(0.5, 0.5, 0.5)

bunnyOG = GetActor(bunnyOG_PD, 0, 0, 1, 5)
bunny1 = GetActor(bunny1_PD, 1, 0, 0, 5)

renderer.AddActor(bunnyOG)
renderer.AddActor(bunny1)

window = vtkRenderWindow()
window.AddRenderer(renderer) #Moguće je dodati i više renderera na jedan prozor
window.SetSize(800, 600) #Veličina prozora na ekranu
window.SetWindowName("Scena") #Naziv prozora

interactor = vtkRenderWindowInteractor()
interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
interactor.SetRenderWindow(window)
interactor.Start()

# new window after closing the first one
renderer = vtkRenderer()
renderer.SetBackground(0.5, 0.5, 0.5)

bunnyOG = GetActor(bunnyOG_PD, 0, 0, 1, 5)
renderer.AddActor(bunnyOG)

window = vtkRenderWindow()
window.AddRenderer(renderer) #Moguće je dodati i više renderera na jedan prozor
window.SetSize(800, 600) #Veličina prozora na ekranu
window.SetWindowName("Scena") #Naziv prozora

interactor = vtkRenderWindowInteractor()
interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
interactor.SetRenderWindow(window)

for i in range(totalIterations):
    icp.SetSource(bunny1_PD) #Ulazni objekt (početna poza objekta)
    icp.SetTarget(bunnyOG_PD) #Konačni objekt (željena poza objekta)
    icp.GetLandmarkTransform().SetModeToRigidBody() #Potrebni način rada je transformacija za kruta tijela
    icp.SetMaximumNumberOfIterations(1) #Željeni broj iteracija
    icp.SetMaximumNumberOfLandmarks(pointPairs) #Koliko parova točaka da se koristi prilikom minimiziranja cost funkcije

    totalTime -= time.time()
    icp.Update()
    totalTime += time.time()

    icpTransformFilter = vtkTransformPolyDataFilter()
    icpTransformFilter.SetInputData(bunny1_PD) #Objekt s početnim koordinatama
    icpTransformFilter.SetTransform(icp) #transformiramo na novi položaj koristeći transformacijsku matricu
    icpTransformFilter.Update()

    bunny1_PD = icpTransformFilter.GetOutput() #Transformirani (novi) objekt



    if(i != 0):
        renderer.RemoveActor(bunny1_corr)
    bunny1_corr = GetActor(bunny1_PD, 0, 1, 0, 5)
    renderer.AddActor(bunny1_corr)
    renderer.ResetCamera() #Centriraj kameru tako da obuhvaća objekte na sceni

    window.Render() #Renderaj scenu

    time.sleep(dt)

print(f"Total time: {totalTime} seconds")
interactor.Start() #Pokretanje interaktora, potrebno kako se vtk prozor ne bi odmah zatvorio


