import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

MODO_PATH = "/home/wx/WorkSpace/IRL/samlle_dog"

QMINI_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{MODO_PATH}/Model/Q1/urdf/q1.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42), 
        joint_pos={
            # --- 左腿 (Left Leg) ---
            # 顺序: Yaw -> Roll -> Pitch -> Knee -> Ankle
            ".*hip_yaw_l": 0.4,
            ".*hip_roll_l": -0.1,
            ".*hip_pitch_l": -1.5,
            ".*knee_pitch_l": 1.0,
            ".*ankle_pitch_l": -1.3,

            # --- 右腿 (Right Leg) ---
            # 注意：基于您提供的参考列表，数值与左腿相反
            ".*hip_yaw_r": -0.4,
            ".*hip_roll_r": 0.1,
            ".*hip_pitch_r": 1.5,
            ".*knee_pitch_r": -1.0,
            ".*ankle_pitch_r": 1.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # 1. Hip Yaw (偏航)
        "hip_yaw": DCMotorCfg(
            joint_names_expr=[".*hip_yaw_l", ".*hip_yaw_r"],
            effort_limit=20.0,
            saturation_effort=20.0,  # [新增] 必须添加此参数，通常等于 effort_limit
            velocity_limit=30.0,
            stiffness=30.0,
            damping=1.0,
            friction=0.0,
        ),

        # 2. Hip Roll (侧展)
        "hip_roll": DCMotorCfg(
            joint_names_expr=[".*hip_roll_l", ".*hip_roll_r"],
            effort_limit=60.0,
            saturation_effort=60.0,  # [新增] 必须添加
            velocity_limit=10.0,
            stiffness=80.0,
            damping=2.0,
            friction=0.00,
        ),

        # 3. Hip Pitch (大腿)
        "hip_pitch": DCMotorCfg(
            joint_names_expr=[".*hip_pitch_l", ".*hip_pitch_r"],
            effort_limit=20.0,
            saturation_effort=20.0,  # [新增] 必须添加
            velocity_limit=30.0,
            stiffness=55.0,
            damping=1.5,
            friction=0.0,
        ),

        # 4. Knee Pitch (膝盖)
        "knee_pitch": DCMotorCfg(
            joint_names_expr=[".*knee_pitch_l", ".*knee_pitch_r"],
            effort_limit=20.0,
            saturation_effort=20.0,  # [新增] 必须添加
            velocity_limit=30.0,
            stiffness=55.0,
            damping=1.5,
            friction=0.0,
        ),

        # 5. Ankle Pitch (脚踝)
        "ankle_pitch": DCMotorCfg(
            joint_names_expr=[".*ankle_pitch_l", ".*ankle_pitch_r"],
            effort_limit=20.0,
            saturation_effort=20.0,  # [新增] 必须添加
            velocity_limit=30.0,
            stiffness=20.0,
            damping=0.25,
            friction=0.0,
        ),
    },
)