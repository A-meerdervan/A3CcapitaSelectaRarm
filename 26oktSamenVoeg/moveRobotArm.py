# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 11:47:13 2018

@author: arnol
"""
from dynamixel_sdk import *                    # Uses Dynamixel SDK library
import numpy as np
import time
import gobalConst as cn

class RobotController:
    def __init__(self, serialPort = 'COM4', _print = True):
        self.maxAngles = cn.rob_MaxJointAngle

        # set the default values for the controller
        # Control table address
        self.ADDR_MX_TORQUE_ENABLE       = 24           # Control table address is different in Dynamixel model
        self.ADDR_MX_GOAL_POSITION       = 30
        self.ADDR_MX_PRESENT_POSITION    = 36
        self.ADDR_MX_GOAL_VELOCITY       = 32
        self.ADDR_MX_PRESENT_VELOCITY    = 38

        # Protocol version
        self.PROTOCOL_VERSION            = 1.0          # See which protocol version is used in the Dynamixel

        # Default setting
        self.BAUDRATE                    = 57600
        self.TORQUE_ENABLE               = 1            # Value for enabling the torque
        self.TORQUE_DISABLE              = 0            # Value for disabling the torque
        self.DXL_INITIAL_POSITION_VALUE1 = 820           # Define center position
        self.DXL_INITIAL_POSITION_VALUE2 = 512
        self.DXL_INITIAL_POSITION_VALUE3 = 516
        self.DXL_VELOCITY_VALUE          = 130
        self.DXL_MOVING_STATUS_THRESHOLD = 5           # Dynamixel moving status threshold

        self.COMM_SUCCESS                = 0            # Communication Success result value
        self.COMM_TX_FAIL                = -3001        # Communication Tx Failed
        self.STEP_SIZE                   = cn.rob_StepSize         # Stepsize in degrees

        self.DXL1_ID = 32
        self.DXL2_ID = 33
        self.DXL3_ID = 34
        self.SERIAL_PORT = serialPort

        self.initCommunication(_print)
        self.hasConnection = self.testCommunicationMotors()
        if not self.hasConnection:
            print('Could not contact motors. Press key to exit, then try again')
            input()
            exit()


    def initCommunication(self, _print):
        # Initialize PortHandler instance
        # Set the port path
        # Get methods and members of PortHandlerLinux or PortHandlerWindows
        self.portHandler = PortHandler(self.SERIAL_PORT)

        # Initialize PacketHandler instance
        # Set the protocol version
        # Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)

        # Open port
        if self.portHandler.openPort():
            if (_print):
                print("Succeeded to open the port")
        else:
            if (_print):
                print("Failed to open the port")
                print("Press any key to terminate...")
            input()
            quit()

#         Set port baudrate
        if self.portHandler.setBaudRate(self.BAUDRATE):
            if (_print):
                print("Succeeded to change the baudrate")
        else:
            if (_print):
                print("Failed to change the baudrate")
                print("Press any key to terminate...")
            input()
            quit()

        return

    def testCommunicationMotors(self):
        for ctr in range(3):
            if (ctr == 0):
                DXL_ID = self.DXL1_ID
            elif (ctr == 1):
                DXL_ID = self.DXL2_ID
            elif (ctr == 2):
                DXL_ID = self.DXL3_ID

        # Enable Dynamixel Torque
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, self.ADDR_MX_TORQUE_ENABLE, self.TORQUE_ENABLE)
        if dxl_comm_result != self.COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            return False
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            return False
        else:
            # Disable Dynamixel Torque
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, self.ADDR_MX_TORQUE_ENABLE, self.TORQUE_DISABLE)
            if dxl_comm_result != self.COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))

        print('tested all motors successfully')

        return True

    def moveInitialPosition(self, initAng):
        for i in range(2):
            self.moveArm(i+1, np.radians([90,0,0]), False)
        
        for i in range(3):
#            j = 3 - i - 1
            print('move ', i, 'to: ', np.degrees(initAng[i]))
            self.moveArm(i, initAng, False)

        return

    def moveArm2(self, act, angles, _print):
#        a = np.zeros((6,1))   # added to convert the output of the agent to the simulation env
#        a[action - 1] = 1
##        action = a
#        a = a.reshape((3, 2))
#        actionMap = [1,-1]
#        act = a * actionMap
#        act = np.max(act, axis=1) + np.min(act, axis=1)

        joint = int(np.argmax(np.abs(act)))

        if (joint == 0):
            DXL_ID = self.DXL1_ID
        elif (joint == 1):
            DXL_ID = self.DXL2_ID
        else:
            DXL_ID = self.DXL3_ID
#        dxl_current_position = self.convertAnglesToMotor(angles).astype(int)

        dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, DXL_ID, self.ADDR_MX_PRESENT_POSITION)

        # The communication failes more often than that it works. 
        # When it fails it just sends a realy high number.
        # This loop tries until it gets one in the acceptable range.
        while (dxl_present_position > 1024):
#            print('re-reading the situation')
            try:
                time.sleep(0.01)
                dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, DXL_ID, self.ADDR_MX_PRESENT_POSITION)
            except:
                break

        dummy = np.array([0,0,0])
        dummy[joint] = dxl_present_position
        dxl_goal_position = dummy + 10 * act * np.array([-1,-1,1])

        newAng = self.convertMotorToAngles(dxl_goal_position)

        if (not self.checkAngle(newAng[joint], joint)):
            print ('angle too large to move ', joint)
            print(np.degrees(newAng), '   ', joint)
            return

#        print('moving arm: ', joint)

        dxl_goal_position = dxl_goal_position.astype(int)

        # Enable Dynamixel Torque
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, self.ADDR_MX_TORQUE_ENABLE, self.TORQUE_ENABLE)
        if dxl_comm_result != self.COMM_SUCCESS:
            if (_print):
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            if (_print):
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            if (_print):
                print("Dynamixel has been successfully connected")
#
        while 1:
            # Write goal position
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, DXL_ID, self.ADDR_MX_GOAL_POSITION, dxl_goal_position[joint])
            if dxl_comm_result != self.COMM_SUCCESS:
                if (_print):
                    print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                if (_print):
                    print("%s" % self.packetHandler.getRxPacketError(dxl_error))

            while 1:
                # Read present position
                dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, DXL_ID, self.ADDR_MX_PRESENT_POSITION)
                if dxl_comm_result != self.COMM_SUCCESS:
                    if (_print):
                        print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    if (_print):
                        print("%s" % self.packetHandler.getRxPacketError(dxl_error))
                if (_print):
                    print("[ID:%03d] GoalPos:%03d  PresPos:%03d" % (DXL_ID, dxl_goal_position[joint], dxl_present_position))

                if not abs(dxl_goal_position[joint] - dxl_present_position) > self.DXL_MOVING_STATUS_THRESHOLD:
                    break

            if not abs(dxl_goal_position[joint] - dxl_present_position) > self.DXL_MOVING_STATUS_THRESHOLD:
                break

        # Disable Dynamixel Torque
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, self.ADDR_MX_TORQUE_ENABLE, self.TORQUE_DISABLE)
        if dxl_comm_result != self.COMM_SUCCESS:
            if (_print):
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            if (_print):
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))

        return




    def moveArm(self, armID, angles, _print):
        if (armID == 0):
            DXL_ID = self.DXL1_ID
        elif (armID == 1):
            DXL_ID = self.DXL2_ID
        else:
            DXL_ID = self.DXL3_ID
        print('moving arm: ', armID)

        if (not self.checkAngle(angles[armID], armID)):
            print ('angle too large')
            print(np.degrees(angles), '   ', armID)
            return

        dxl_goal_position = self.convertAnglesToMotor(angles).astype(int)

        # Enable Dynamixel Torque
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, self.ADDR_MX_TORQUE_ENABLE, self.TORQUE_ENABLE)
        if dxl_comm_result != self.COMM_SUCCESS:
            if (_print):
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            if (_print):
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            if (_print):
                print("Dynamixel has been successfully connected")
#
        while 1:
            # Write goal position
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, DXL_ID, self.ADDR_MX_GOAL_POSITION, dxl_goal_position[armID])
            if dxl_comm_result != self.COMM_SUCCESS:
                if (_print):
                    print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                if (_print):
                    print("%s" % self.packetHandler.getRxPacketError(dxl_error))

            while 1:
                # Read present position
                dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, DXL_ID, self.ADDR_MX_PRESENT_POSITION)
                if dxl_comm_result != self.COMM_SUCCESS:
                    if (_print):
                        print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    if (_print):
                        print("%s" % self.packetHandler.getRxPacketError(dxl_error))
                if (_print):
                    print("[ID:%03d] GoalPos:%03d  PresPos:%03d" % (DXL_ID, dxl_goal_position[armID], dxl_present_position))

                if not abs(dxl_goal_position[armID] - dxl_present_position) > self.DXL_MOVING_STATUS_THRESHOLD:
                    break

            if not abs(dxl_goal_position[armID] - dxl_present_position) > self.DXL_MOVING_STATUS_THRESHOLD:
                break

        # Disable Dynamixel Torque
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, self.ADDR_MX_TORQUE_ENABLE, self.TORQUE_DISABLE)
        if dxl_comm_result != self.COMM_SUCCESS:
            if (_print):
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            if (_print):
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))


    def checkAngle(self, a, ID):
        if (a >= self.maxAngles[ID*2] or a <= self.maxAngles[ID*2 + 1]):
            return False

        return True

    def checkAngles(self, angles):
        ctr = 0
        for a in angles:
            if (a >= self.maxAngles[ctr*2] or a <= self.maxAngles[ctr*2 + 1]):
                return False
            ctr += 1

        return True

    def convertAnglesToMotor(self, jointAngles):
        d2m = 512 / 150   # conversion from degrees to motor units

        init = np.array([self.DXL_INITIAL_POSITION_VALUE1, self.DXL_INITIAL_POSITION_VALUE2,
                         self.DXL_INITIAL_POSITION_VALUE3])
        motorUnits = np.array([-1,-1,1]) * np.degrees(jointAngles) * d2m + init


        return motorUnits

    def convertMotorToAngles(self, motorAngles):
        m2d = 150 / 512;   # conversion from motor units to degrees

        init = np.array([self.DXL_INITIAL_POSITION_VALUE1, self.DXL_INITIAL_POSITION_VALUE2,
                     self.DXL_INITIAL_POSITION_VALUE3])

        angles = (motorAngles - init) * m2d *  np.array([-1,-1,1])
        angles = np.radians(angles)

        return angles

