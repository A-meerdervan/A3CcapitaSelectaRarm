# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 11:47:13 2018

@author: arnol
"""
from dynamixel_sdk import *                    # Uses Dynamixel SDK library
import numpy as np
import time

class RobotController:
    def __init__(self, serialPort = 'COM1', _print = True):
        self.maxAngles = np.radians([[0,180], [-90,90], [-90,90]])

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
        self.DXL_INITIAL_POSITION_VALUE1 = 650 - 307    # Define center position
        self.DXL_INITIAL_POSITION_VALUE2 = 800
        self.DXL_INITIAL_POSITION_VALUE3 = 630
        self.DXL_VELOCITY_VALUE          = 110
        self.DXL_MOVING_STATUS_THRESHOLD = 10           # Dynamixel moving status threshold

#        self.ESC_CHARACTER               = 'e'          # Key for escaping loop

        self.COMM_SUCCESS                = 0            # Communication Success result value
        self.COMM_TX_FAIL                = -3001        # Communication Tx Failed

        self.DXL1_ID = 32 #test
        self.DXL2_ID = 33
        self.DXL3_ID = 34
        self.SERIAL_PORT = serialPort

        self.initCommunication(_print)


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
#            getch()
            quit()

#         Set port baudrate
        if self.portHandler.setBaudRate(self.BAUDRATE):
            if (_print):
                print("Succeeded to change the baudrate")
        else:
            if (_print):
                print("Failed to change the baudrate")
                print("Press any key to terminate...")
#            getch()d
            quit()

        return

    def moveArm(self, armID, angles, _print):
        if (armID == 0):
            DXL_ID = self.DXL1_ID
        elif (armID == 1):
            DXL_ID = self.DXL2_ID
        else:
            DXL_ID = self.DXL3_ID

        if (not self.checkAngles(angles)):
            return

        dxl_goal_position = self.convertAnglesToMotor(angles).astype(int)
        print(DXL_ID)
        # Enable Dynamixel Torque
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, self.ADDR_MX_TORQUE_ENABLE, self.TORQUE_ENABLE)
        if dxl_comm_result != self.COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
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
                    print('got there')
                    break

        # Disable Dynamixel Torque
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, self.ADDR_MX_TORQUE_ENABLE, self.TORQUE_DISABLE)
        if dxl_comm_result != self.COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))

    def readCurrentPosition(self, armID, angles):
        if (armID == 0):
            DXL_ID = self.DXL1_ID
        elif (armID == 1):
            DXL_ID = self.DXL2_ID
        else:
            DXL_ID = self.DXL3_ID

        dxl_goal_position = self.convertAnglesToMotor(angles).astype(int)
        print(dxl_goal_position)
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, self.ADDR_MX_TORQUE_ENABLE, self.TORQUE_ENABLE)
        if dxl_comm_result != self.COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel has been successfully connected")

        ctr = 0
        while 1:
            # Read present position
            dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, DXL_ID, self.ADDR_MX_PRESENT_POSITION)
#            time.sleep(1)
            print(dxl_comm_result)
            if dxl_comm_result != self.COMM_SUCCESS:
                if (_print):
                    print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                if (_print):
                    print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            if (_print):
                print("[ID:%03d] GoalPos:%03d  PresPos:%03d" % (DXL_ID, dxl_goal_position[armID], dxl_present_position))

            ctr += 1

            if ctr > 2:
                break

        # Disable Dynamixel Torque
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, self.ADDR_MX_TORQUE_ENABLE, self.TORQUE_DISABLE)
        if dxl_comm_result != self.COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))

        time.sleep(5)
#        quit()
        return


    def checkAngles(self, angles):
        ctr = 0
        for a in angles:
            if (a < self.maxAngles[ctr, 0] or a > self.maxAngles[ctr, 1]):
                return False
            ctr += 1

        return True

    def convertAnglesToMotor(self, jointAngles):
        d2m = 512 / 150   # conversion from degrees to motor units

        init = np.array([self.DXL_INITIAL_POSITION_VALUE1, self.DXL_INITIAL_POSITION_VALUE2,
                         self.DXL_INITIAL_POSITION_VALUE3])
        motorUnits = np.degrees(jointAngles) * d2m + init


        return motorUnits

    def convertMotorToAngles(self, motorAngles):
        m2d = 150 / 512;   # conversion from motor units to degrees

        init = np.array([self.DXL_INITIAL_POSITION_VALUE1, self.DXL_INITIAL_POSITION_VALUE2,
                     self.DXL_INITIAL_POSITION_VALUE3])

        angles = (motorAngles - init) * m2d
        angles = np.radians(angles)

        return angles


_print = True
controller = RobotController('COM4', _print)

angles = np.radians(np.array([90,10,-10]))

controller.moveArm(1, angles, _print)
#controller.readCurrentPosition(0, angles)