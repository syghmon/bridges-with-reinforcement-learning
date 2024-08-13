from compas_fab.backends import PyBulletClient
from compas_fab.utilities import LazyLoader

pybullet = LazyLoader('pybullet', globals(), 'pybullet')


class CompasClient(PyBulletClient):
    """
    Create a compas_pybullet for inverse kinematics calculation
    """
    JOINT_FIXED = pybullet.JOINT_FIXED
    


    def resetSimulation(self):
        return pybullet.resetSimulation(physicsClientId=self.client_id)

    def loadURDF(self, path, pos, orientation, useFixedBase=0, flags=0):
        return pybullet.loadURDF(path, pos, orientation, useFixedBase=useFixedBase, physicsClientId=self.client_id, flags=flags)

    def setGravity(self, x, y, z):
        return pybullet.setGravity(x, y, z, physicsClientId=self.client_id)

    def setTimeStep(self, timeStep):
        return pybullet.setTimeStep(timeStep, physicsClientId=self.client_id)

    def setRealTimeSimulation(self, num):
        return pybullet.setRealTimeSimulation(num, physicsClientId=self.client_id)

    def removeBody(self, object_id):
        return pybullet.removeBody(object_id, physicsClientId=self.client_id)

    def getBasePositionAndOrientation(self, object_id):
        return pybullet.getBasePositionAndOrientation(object_id, physicsClientId=self.client_id)

    def getContactPoints(self, bodyA, bodyB):
        return pybullet.getContactPoints(bodyA, bodyB, physicsClientId=self.client_id)

    def performCollisionDetection(self):
        return pybullet.performCollisionDetection(physicsClientId=self.client_id)

    def resetDebugVisualizerCamera(self, cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition):
        return pybullet.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition, physicsClientId=self.client_id)
    
    def createConstraint(self, **kwargs):
        return pybullet.createConstraint(physicsClientId=self.client_id, **kwargs)
    
    def removeConstraint(self, constraint_id):
        return pybullet.removeConstraint(constraint_id, physicsClientId=self.client_id)

    def change_object_color(self, object_id, color, link_index=-1):
        pybullet.changeVisualShape(object_id, link_index, rgbaColor=color, physicsClientId=self.client_id)
    
    def changeDynamics(self, body_id, link_id, **kwargs):
        return pybullet.changeDynamics(body_id, link_id, physicsClientId=self.client_id, **kwargs)