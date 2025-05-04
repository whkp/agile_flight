#!/usr/bin/python3
import argparse

import rospy
from dodgeros_msgs.msg import Command
from dodgeros_msgs.msg import QuadState
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Empty

from envsim_msgs.msg import ObstacleArray
from rl_example import load_rl_policy
from user_code import compute_command_vision_based, compute_command_state_based
from utils import AgileCommandMode, AgileQuadState
from VisionController import VisionBasedController, init_controller

import cv2
from copy import deepcopy
import numpy as np
import pandas as pd
import time
import os


class AgilePilotNode:
    def __init__(self, vision_based=False, desVel=None, ppo_path=None):
        print("Initializing agile_pilot_node...")
        rospy.init_node('agile_pilot_node', anonymous=False)

        self.vision_based = vision_based
        self.rl_policy = None
        if ppo_path is not None:
            self.rl_policy = load_rl_policy(ppo_path)
        self.publish_commands = False
        self.cv_bridge = CvBridge()
        self.state = None
        self.last_valid_img = None
        self.rgb_img = None
        self.desiredVel = desVel

        self.start_time = 0
        self.init = 0
        self.t1 = 0
        self.col = None
        self.data_recorded = True
        self.data_collection_xrange = [2, 60]
        self.time_interval = .03
        self.count = 0
        self.depth_im_threshold = 0.09

        #自定义从轨迹到控制指令生成
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_drone_navigation_model.pth')
        trajectory_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'selected_indices_yaw.npy')
        self.controller = VisionBasedController(model_path, trajectory_path)

        data_log_format = {'timestamp':[],
                           'desired_vel':[],
                           'quat_1':[],
                           'quat_2':[],
                           'quat_3':[],
                           'quat_4':[],
                           'pos_x':[],
                           'pos_y':[],
                           'pos_z':[],
                           'vel_x':[],
                           'vel_y':[],
                           'vel_z':[],
                           'acc_x':[],
                           'acc_y':[],
                           'acc_z':[],
                           'velcmd_x':[],
                           'velcmd_y':[],
                           'velcmd_z':[],
                           'ct_cmd':[],
                           'br_cmd_x':[],
                           'br_cmd_y':[],
                           'br_cmd_z':[],
                           'is_collide': [],
        } 
        self.data_log = pd.DataFrame(data_log_format)
        self.folder = f"train_set/{int(time.time()*100)}" 
        if(self.data_recorded == True):
            os.makedirs(self.folder)

        self.logged_time_flag = 0

        quad_name = 'kingfisher'

        # Logic subscribers
        self.start_sub = rospy.Subscriber("/" + quad_name + "/start_navigation", Empty, self.start_callback,
                                          queue_size=1, tcp_nodelay=True)

        # Observation subscribers
        self.odom_sub = rospy.Subscriber("/" + quad_name + "/dodgeros_pilot/state", QuadState, self.state_callback,
                                         queue_size=1, tcp_nodelay=True)

        self.img_sub = rospy.Subscriber("/" + quad_name + "/dodgeros_pilot/unity/depth", Image, self.img_callback,
                                        queue_size=1, tcp_nodelay=True)
        self.obstacle_sub = rospy.Subscriber("/" + quad_name + "/dodgeros_pilot/groundtruth/obstacles", ObstacleArray,
                                             self.obstacle_callback, queue_size=1, tcp_nodelay=True)
        self.rgb_img_sub = rospy.Subscriber("/" + quad_name + "/dodgeros_pilot/unity/image", 
                                            Image, 
                                            self.rgb_img_callback,
                                            queue_size=1, 
                                            tcp_nodelay=True)
        # Command publishers
        self.cmd_pub = rospy.Publisher("/" + quad_name + "/dodgeros_pilot/feedthrough_command", Command, queue_size=1)
        self.linvel_pub = rospy.Publisher("/" + quad_name + "/dodgeros_pilot/velocity_command", TwistStamped,
                                          queue_size=1)
        self.debug_img1_pub = rospy.Publisher("/" + quad_name + "/dodgeros_pilot/debug_img1", Image, queue_size=1)
        self.debug_img2_pub = rospy.Publisher("/" + quad_name + "/dodgeros_pilot/debug_img2", Image, queue_size=1)
        print("Initialization completed!")

    def img_callback(self, img_data):
        if self.controller is None:
            try:
                init_controller()
            except Exception as e:
                rospy.logerr(f"Failed to initialize controller: {str(e)}")
                return

        img = self.cv_bridge.imgmsg_to_cv2(img_data, desired_encoding='passthrough')
        # 将深度图像归一化到[0, 1]
        img = np.clip(img/self.depth_im_threshold, 0, 1)
        #跟踪有效的图像数据
        self.last_valid_img = deepcopy(img) if img.min() > 0.0 else self.last_valid_img

        if not self.vision_based:
            return
        if self.state is None:
            return
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_data, desired_encoding='passthrough')
        command = compute_command_vision_based(self.state, cv_image, controller=self.controller, desiredVel = self.desiredVel)
        self.publish_command(command)

    def rgb_img_callback(self, img_data):
        self.rgb_img = self.cv_bridge.imgmsg_to_cv2(img_data, desired_encoding='passthrough')


    def state_callback(self, state_data):
        self.state = AgileQuadState(state_data, self.desiredVel)

    def obstacle_callback(self, obs_data):
        if self.vision_based:
            return
        if self.state is None:
            return
        command = compute_command_state_based(state=self.state, obstacles=obs_data, desiredVel = self.desiredVel, rl_policy=self.rl_policy)
        self.publish_command(command)
        if self.state.pos[0] < 0.1:
            self.start_time = command.t
        if self.data_recorded and self.state.pos[0] >= 60 and self.logged_time_flag == 0:
            file = "timeTaken.dat"
            with open(file, "a") as file:
                file.write(str(float(command.t - self.start_time))+"\n")
            self.logged_time_flag = 1
        
        # if we exceed the time interval then save the data
        if self.data_recorded and (self.state.t - self.t1 > self.time_interval or self.t1 == 0 or self.col) and (self.state.pos[2] > 2.95 or self.init == 1):
            
            self.init = 1

            if self.state.pos[0] > self.data_collection_xrange[0] and self.state.pos[0] < self.data_collection_xrange[1]:

                # reset the time flag
                self.t1 = self.state.t

                # Get the current time stamp
                timestamp = round(self.state.t, 3)  # If you need more hz, you might need to modify this round

                # Save the image by the name of that instant
                # np.save(self.folder + f"/im_{timestamp}", self.last_valid_img)
                #只保存深度图像
                cv2.imwrite(f"{self.folder}/{str(timestamp)}.png", (self.last_valid_img*255).astype(np.uint8))
                #cv2.imwrite(f"{self.folder}/{str(timestamp)}_rgb.png", (self.rgb_img*255).astype(np.uint8))

                # Get the collision flag
                col = self.if_collide(obs_data.obstacles[0])
                # Append the data frame
                # @TODO: This needs to be managed better if the number of datapoints exceeds 10,000
                self.data_log.loc[len(self.data_log)] = [
                    timestamp,
                    self.desiredVel,
                    self.state.att[0],
                    self.state.att[1],
                    self.state.att[2],
                    self.state.att[3],
                    self.state.pos[0],
                    self.state.pos[1],
                    self.state.pos[2],
                    self.state.vel[0],
                    self.state.vel[1],
                    self.state.vel[2],
                    self.state.acc[0],
                    self.state.acc[1],
                    self.state.acc[2],
                    command.velocity[0],
                    command.velocity[1],
                    command.velocity[2],
                    self.curr_cmd.collective_thrust,
                    self.curr_cmd.bodyrates.x,
                    self.curr_cmd.bodyrates.y,
                    self.curr_cmd.bodyrates.z,
                    self.col,
                ]

                # Counter flag for saving the data frame
                self.count += 1

        # Save once every 10 instances - writing every instance can be expensive
        if self.data_recorded and self.count % 2 == 0 and self.count != 0 or abs(self.state.pos[0] - 20) < 1:
            self.data_log.to_csv(self.folder + "/data.csv")

    def publish_command(self, command):
        if command.mode == AgileCommandMode.SRT:
            assert len(command.rotor_thrusts) == 4
            cmd_msg = Command()
            cmd_msg.t = command.t
            cmd_msg.header.stamp = rospy.Time(command.t)
            cmd_msg.is_single_rotor_thrust = True
            cmd_msg.thrusts = command.rotor_thrusts
            if self.publish_commands:
                self.cmd_pub.publish(cmd_msg)
                return
        elif command.mode == AgileCommandMode.CTBR:
            assert len(command.bodyrates) == 3
            cmd_msg = Command()
            cmd_msg.t = command.t
            cmd_msg.header.stamp = rospy.Time(command.t)
            cmd_msg.is_single_rotor_thrust = False
            cmd_msg.collective_thrust = command.collective_thrust
            cmd_msg.bodyrates.x = command.bodyrates[0]
            cmd_msg.bodyrates.y = command.bodyrates[1]
            cmd_msg.bodyrates.z = command.bodyrates[2]
            if self.publish_commands:
                self.cmd_pub.publish(cmd_msg)
                return
        elif command.mode == AgileCommandMode.LINVEL:
            vel_msg = TwistStamped()
            vel_msg.header.stamp = rospy.Time(command.t)
            vel_msg.twist.linear.x = command.velocity[0]
            vel_msg.twist.linear.y = command.velocity[1]
            vel_msg.twist.linear.z = command.velocity[2]
            vel_msg.twist.angular.x = 0.0
            vel_msg.twist.angular.y = 0.0
            vel_msg.twist.angular.z = command.yawrate
            if self.publish_commands:
                self.linvel_pub.publish(vel_msg)
                return
        else:
            assert False, "Unknown command mode specified"

    def start_callback(self, data):
        print("Start publishing commands!")
        self.publish_commands = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Agile Pilot.')
    parser.add_argument('--vision_based', help='Fly vision-based', required=False, dest='vision_based',
                        action='store_false')
    parser.add_argument('--desVel', type=float, help='Desired velocity', default=5.0)
    parser.add_argument('--ppo_path', help='PPO neural network policy', required=False,  default=None)

    args = parser.parse_args()
    agile_pilot_node = AgilePilotNode(vision_based=args.vision_based, desVel= args.desVel, ppo_path=args.ppo_path)
    rospy.spin()
