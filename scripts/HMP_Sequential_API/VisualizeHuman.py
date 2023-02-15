from config import model_path, file_name, jointOrder
#import idyntree.bindings as iDynTree
from idyntree.bindings import ModelLoader
from idyntree.bindings import KinDynComputations
from idyntree.bindings import Vector4
from idyntree.bindings import Position
from idyntree.bindings import Rotation
from idyntree.bindings import Transform
from idyntree.bindings import Direction
from idyntree.bindings import VectorDynSize
from idyntree.bindings import VisualizerOptions
from idyntree.bindings import Visualizer
import time

import numpy as np


def visualize_human():

    mdlLoader = ModelLoader()
    dynComp = KinDynComputations()
    urdf_file = model_path + file_name

    mdlLoader.loadModelFromFile(urdf_file)
    dynComp.loadRobotModel(mdlLoader.model())
    print
    "The loaded model has", dynComp.model().getNrOfDOFs(), \
    "internal degrees of freedom and", dynComp.model().getNrOfLinks(), "links."

    model = mdlLoader.model()
    dofs = model.getNrOfDOFs()
    base_link = 'RightFoot'
    joints_positions = np.zeros((dofs, 1))

    model.setDefaultBaseLink(model.getLinkIndex(base_link))

    wHb = Transform.Identity()
    joints = VectorDynSize(model.getNrOfDOFs())
    joints.zero()

    cameraDeltaPosition = Position(3.0, 0.0, 1.0)
    fixedCameraTarget = Position(0.0, 0.0, 0.0)

    viz = Visualizer()
    options = VisualizerOptions()
    viz.init(options)
    viz.camera().setPosition(cameraDeltaPosition)
    viz.camera().setTarget(fixedCameraTarget)
    # viz.camera().animator().enableMouseControl(True)
    viz.addModel(model, "human")

    base_position = Position(0, 0, 1.0)
    wHb.setPosition(base_position)
    wHb.setRotation(Rotation.Identity())
    viz.modelViz("human").setPositions(wHb, joints)
    time.sleep(1)
    viz.draw()

    time.sleep(10)

    # world_H_base = np.array([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 1, 0.6], [0, 0, 0, 1]])
    # dynComp.setRobotState(world_H_base, joints_positions)
    # print('visualize_human()')
    return 0
