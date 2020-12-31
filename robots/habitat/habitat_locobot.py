import numpy as np
import math
import pyrobot.utils.util as prutil
import rospy

import habitat_sim
import habitat_sim.agent as habAgent
import habitat_sim.utils as habUtils
from habitat_sim.agent.controls import ActuationSpec
import habitat_sim.bindings as hsim
import habitat_sim.errors

import quaternion
from tf.transformations import euler_from_quaternion, euler_from_matrix

#################
# util function #
#################

# build SimulatorConfiguration
def make_cfg(SIM):
    sim_cfg = hsim.SimulatorConfiguration()

    if SIM.SCENE_ID == "none":
        SIM.SCENE_ID = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
    sim_cfg.scene.id = SIM.SCENE_ID

    sim_cfg.enable_physics = SIM.PHYSICS
    if SIM.PHYSICS:
        sim_cfg.physics_config_file = SIM.PHYSICS_CONFIG_FILE
    # sim_cfg.gpu_device_id = 0
    # sim_cfg.scene.id = settings["scene"]

    # define default sensor parameters (see src/esp/Sensor/Sensor.h)
    sensors = dict()
    for i in range(len(SIM.AGENT.SENSORS.NAMES)):
        sensors[SIM.AGENT.SENSORS.NAMES[i]] = {
            "sensor_type": getattr(hsim.SensorType, SIM.AGENT.SENSORS.TYPES[i]),
            "resolution": [
                SIM.AGENT.SENSORS.RESOLUTIONS[i][0],
                SIM.AGENT.SENSORS.RESOLUTIONS[i][1],
            ],
            "position": [
                SIM.AGENT.SENSORS.POSES[i][0],
                SIM.AGENT.SENSORS.POSES[i][1],
                SIM.AGENT.SENSORS.POSES[i][2],
            ],
            "orientation": [
                SIM.AGENT.SENSORS.POSES[i][3],
                SIM.AGENT.SENSORS.POSES[i][4],
                SIM.AGENT.SENSORS.POSES[i][5],
            ],
        }

    # create sensor specifications
    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        sensor_spec = hsim.SensorSpec()
        sensor_spec.uuid = sensor_uuid
        sensor_spec.sensor_type = sensor_params["sensor_type"]
        sensor_spec.resolution = sensor_params["resolution"]
        sensor_spec.position = sensor_params["position"]
        sensor_spec.gpu2gpu_transfer = False  # Todo: Move this to config

        print("==== Initialized Sensor Spec: =====")
        print("Sensor uuid: ", sensor_spec.uuid)
        print("Sensor type: ", sensor_spec.sensor_type)
        print("Sensor position: ", sensor_spec.position)
        print("===================================")

        sensor_specs.append(sensor_spec)

    # create agent specifications
    # TODO: Accomodate more agents
    agent_cfg = habAgent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    # TODO: Move agent actions to config
    agent_cfg.action_space = {
        "move_forward": habAgent.ActionSpec(
            "move_forward", habAgent.ActuationSpec(amount=1.0)
        ),
        "turn_left": habAgent.ActionSpec(
            "turn_left", habAgent.ActuationSpec(amount=10.0)
        ),
        "turn_right": habAgent.ActionSpec(
            "turn_right", habAgent.ActuationSpec(amount=10.0)
        ),
    }
    sim_cfg.default_agent_id = SIM.DEFAULT_AGENT_ID
    # # override action space to no-op to test physics
    # if sim_cfg.enable_physics:
    #     agent_cfg.action_space = {
    #         "move_forward": habAgent.ActionSpec(
    #             "move_forward", habAgent.ActuationSpec(amount=0.0)
    #         )
    #     }
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

class HabitatLocobot:
	def __init__(self, configs):
		rospy.init_node("habitat_locobot")

		self.configs = configs

		self.sim_config = copy.deepcopy(self.configs.COMMON.SIMULATOR)
		self.sim_config.defrost()
		self.sim = habitat_sim.Simulator(make_cfg(self.sim_config))
		print(self.sim_config)

		self.agent = self.sim.get_agent(self.configs.COMMON.SIMULATOR.DEFAULT_AGENT_ID)

		self.ctrl_rate = 100
		self.realtime_scale_factor = 10
		self.interval = rospy.Duration.from_sec(1/self.ctrl_rate)
		self.clock_pub = rospy.Publisher('/clock', Clock, queue_size=1)
		self.sim_time = rospy.Time()
		clock = Clock()
		self.clock_pub.publish(clock)

		self.img_rate = 25
		self.cv_bridge = CvBridge()
		self.rgb_img = None
		self.rgb_pub = rospy.Publisher('/camera/color/image_raw', Image, queue_size=1)

		self.lin_speed = 0.
		self.ang_speed = 0.

		rospy.Subscriber(
			"/test_base_vel",
			Twist,
			self._command_callback,
		)

		self.vel_control = habitat_sim.physics.VelocityControl()
		self.vel_control.controlling_lin_vel = True
		self.vel_control.lin_vel_is_local = True
		self.vel_control.controlling_ang_vel = True
		self.vel_control.ang_vel_is_local = True


	def _command_callback(self, msg):
		self.lin_speed = msg.linear.x
		self.ang_speed = msg.angular.z

	def base_step(self):
		self.vel_control.linear_velocity = np.array([0,0,-self.lin_speed])
		self.vel_control.angular_velocity = np.array([0,self.ang_speed,0])

		state = self.agent.state
		previous_rigid_state = habitat_sim.RigidState(utils.quat_to_magnum(state.rotation), state.position)
		target_rigid_state = self.vel_control.integrate_transform(1/self.ctrl_rate, previous_rigid_state)
		end_pos = self.sim.step_filter(previous_rigid_state.translation, target_rigid_state.translation)
		state.position = end_pos
		state.rotation = utils.quat_from_magnum(target_rigid_state.rotation)
		self.agent.set_state(state)

		dist_moved_before_filter = (
			target_rigid_state.translation - previous_rigid_state.translation
		).dot()
		dist_moved_after_filter = (
			end_pos - previous_rigid_state.translation
		).dot()
		EPS = 1e-5
		collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter
		self.sim.step_physics(1/self.ctrl_rate)

		self.sim_time = self.sim_time + self.interval
		clock = Clock()
		clock.clock = self.sim_time
		self.clock_pub.publish(clock)
		# self.visualize()
		base_state = self.base.get_state()
		print("Base State:")
		print(base_state)

	def img_step(self):
		self.rgb_img = self.bot.camera.get_rgb()
		print("Broadcast Image:")
		print(self.rgb_img is not None)
		try:
			rgb_img_msg = self.cv_bridge.cv2_to_imgmsg(self.rgb_img, encoding="bgr8")
			self.rgb_pub.publish(rgb_img_msg)
		except Exception as e:
			rospy.logerr(e)

	def spin(self):
		prev_time = time.time()
		while not rospy.is_shutdown():
			self.base_step()
			if time.time() - prev_time > 1/(self.img_rate * self.realtime_scale_factor):
				self.img_step()
			prev_time = time.time()
			time.sleep(1/(self.ctrl_rate * self.realtime_scale_factor))

