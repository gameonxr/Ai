# Universal AI Body Simulator / Embodied AI Platform
## Phase 1: Core Architecture & Foundation

---

## EXECUTIVE SUMMARY

**GOAL:** Build a reusable AI agent body/physics simulator that is completely brain-agnostic.

**CORE PRINCIPLE:** The AI brain must be completely separate from physics, body, sensors, actuators, world, and rendering.

**KEY REQUIREMENT:** You must be able to replace the AI brain later without rewriting ANY simulator code.

**THE LOOP:**
```
OBSERVE → THINK → ACT → PHYSICS → OBSERVE
```

**ARCHITECTURE FLOW:**
```
WORLD
  ↓
PHYSICS ENGINE
  ↓
BODY (Humanoid)
  ↓
SENSORS (Proprioception, Vision, IMU)
  ↓
OBSERVATION API
  ↓
AI BRAIN (Future: LLM, RL Agent, etc.)
  ↓
ACTION API
  ↓
ACTUATORS (Joint Motors)
  ↓
PHYSICS ENGINE
  ↓
WORLD
```

---

## CRITICAL ARCHITECTURAL RULE

**THE AI BRAIN MUST BE COMPLETELY SEPARATE FROM:**
- Physics engine internals
- Body structure
- Sensor implementations
- Actuator mechanics
- World/environment
- Renderer (when added)
- Training system (when added)

**All communication must happen through well-defined APIs:**
- Observation API (simulator → brain)
- Action API (brain → simulator)
- No direct access to internal simulator state
- No brain logic in physics engine
- No physics logic in brain interface

---

## TECH STACK (FIXED FOR PHASE 1)

**Language:** Python 3.10+

**Physics Engine (Choose One):**
- Primary: MuJoCo (mature, deterministic, RL-friendly)
- Fallback: PyBullet (open-source, free)

**Configuration:** YAML (human-readable, versionable)

**Testing:** pytest

**Dependencies (Keep Minimal):**
- numpy (math)
- pyyaml (config)
- physics library (mujoco or pybullet)
- pytest (testing)

**Hardware:**
- CPU: Intel/AMD x64 (required)
- GPU: Optional (not needed for Phase 1)
- Memory: 2GB+ recommended

**Platform Support:** Linux, macOS, Windows (all supported)

---

## PHASE 1 DELIVERABLES (MUST HAVE)

### ✅ MANDATORY COMPONENTS

1. **Brain API** (Abstract Interface - No Implementation)
   - Must be implementation-agnostic
   - Support any future brain type
   - Clear contract for brain-simulator interaction

2. **Humanoid Body** (Configurable)
   - 12-15 degrees of freedom (DOF)
   - Limbs: head, neck, torso, upper arms, lower arms, hands, upper legs, lower legs, feet
   - Fully configurable via YAML (no hardcoding)
   - Support mass, inertia, dimensions

3. **Physics Abstraction** (NOT Direct Engine Access)
   - Encapsulate MuJoCo/Bullet
   - AI never touches physics internals directly
   - Only expose safe high-level actions

4. **Sensor System** (Modular Framework)
   - Proprioception (joint positions, velocities, accelerations)
   - Vision (camera stub - placeholder for Phase 1)
   - IMU (acceleration, angular velocity, orientation)
   - Touch (contact detection stub)
   - Extensible for future sensors

5. **Actuator System** (Validated Commands)
   - Joint motors (position and torque control)
   - Configurable limits (force, torque, velocity)
   - Action validation before physics execution
   - Motor response time & damping

6. **Observation Standardization**
   - Unified observation object
   - Only expose configured sensors
   - No hidden simulator information
   - Timestamp included

7. **Action Standardization**
   - Unified action object format
   - Validation (NaN, infinity, limits)
   - Clamping of out-of-range values
   - Rejection of invalid commands with logging

8. **Environment** (Basic)
   - Floor (static ground plane)
   - Gravity (configurable, default: -9.81)
   - One humanoid body
   - Simple collision system

9. **Main Simulation Loop**
   - Fixed timestep (default: 0.005 seconds = 200 Hz)
   - Step function: sim.step() or sim.step(n_steps)
   - Deterministic execution (with seed control)
   - Pause/resume capability
   - Single-step execution for debugging

10. **Configuration System**
    - YAML-based configuration
    - Body configuration (YAML)
    - Physics configuration (YAML)
    - Sensor configuration (YAML)
    - Actuator configuration (YAML)
    - Validation on startup
    - No hardcoded values for configurable parameters

---

### ❌ NOT PHASE 1 (DO NOT BUILD YET)

- Visual rendering/graphics
- RL training framework
- Multi-agent support
- Plugin system (beyond basic architecture)
- Recording/replay system
- Real robot adapters
- Parallel environments
- Audio system
- Procedural world generation
- Task system
- External AI connection (WebSocket, IPC)

**These will be added in Phase 2+ without requiring simulator rewrites.**

---

## PROJECT STRUCTURE

```
ai_body_simulator/
├── README.md
├── requirements.txt
├── setup.py
│
├── config/
│   ├── body_humanoid.yaml         # Body structure definition
│   ├── physics_config.yaml        # Physics parameters
│   ├── sensors_config.yaml        # Sensor configuration
│   ├── actuators_config.yaml      # Actuator configuration
│   └── simulator_config.yaml      # Global simulator config
│
├── simulator/
│   ├── __init__.py
│   ├── simulator.py               # Main simulator class & loop
│   ├── config_loader.py           # Load & validate YAML configs
│   └── constants.py               # Global constants
│
├── brain/
│   ├── __init__.py
│   ├── brain_interface.py         # Abstract brain API
│   ├── dummy_brain.py             # Dummy/test brain implementation
│   └── brain_adapter.py           # Brain ↔ Simulator bridge
│
├── core/
│   ├── __init__.py
│   ├── observation.py             # Standardized observation object
│   ├── action.py                  # Standardized action object
│   ├── validator.py               # Action validation & safety
│   └── types.py                   # Type definitions
│
├── body/
│   ├── __init__.py
│   ├── body.py                    # Body class
│   ├── body_loader.py             # Load body from config
│   ├── joint.py                   # Joint representation
│   └── link.py                    # Link/segment representation
│
├── physics/
│   ├── __init__.py
│   ├── physics_engine.py          # Physics abstraction (interface)
│   ├── mujoco_backend.py          # MuJoCo implementation
│   ├── bullet_backend.py          # PyBullet implementation (fallback)
│   └── physics_utils.py           # Utility functions
│
├── sensors/
│   ├── __init__.py
│   ├── sensor_base.py             # Abstract sensor class
│   ├── sensor_registry.py         # Sensor discovery
│   ├── proprioception.py          # Joint position/velocity
│   ├── vision.py                  # Camera (stub)
│   ├── imu.py                     # Inertial measurement unit
│   ├── touch.py                   # Contact detection (stub)
│   └── observation_builder.py     # Assemble observation from sensors
│
├── actuators/
│   ├── __init__.py
│   ├── actuator_base.py           # Abstract actuator class
│   ├── actuator_registry.py       # Actuator discovery
│   ├── motor.py                   # Joint motor implementation
│   └── actuator_controller.py     # Manage all actuators
│
├── environment/
│   ├── __init__.py
│   ├── world.py                   # World/environment container
│   ├── static_objects.py          # Floor, walls, etc. (Phase 2)
│   └── object_loader.py           # Load objects from config (Phase 2)
│
├── tests/
│   ├── __init__.py
│   ├── test_simulator.py
│   ├── test_brain_api.py
│   ├── test_observation.py
│   ├── test_action.py
│   ├── test_body_loader.py
│   ├── test_physics.py
│   ├── test_sensors.py
│   ├── test_actuators.py
│   ├── test_validator.py
│   └── fixtures.py                # Pytest fixtures
│
└── examples/
    ├── basic_simulation.py        # Minimal working example
    ├── dummy_brain_example.py     # Simple brain implementation
    └── config_examples/           # Example configurations
```

---

## COMPONENT SPECIFICATIONS

### 1. BRAIN API (Most Important - Abstract Interface)

```python
# brain/brain_interface.py

class BrainInterface:
    """
    Abstract interface for any AI brain.
    
    Simulator NEVER knows what type of brain this is.
    Brain NEVER knows about physics/rendering/body internals.
    
    All communication: Observation → Decision → Action
    """
    
    def reset(self, context: dict = None):
        """
        Called at episode start.
        Optional context: {"user_id": "...", "task": "..."}
        """
        pass
    
    def observe(self, observation: 'Observation'):
        """
        Receive sensor data from simulator.
        Observation is standardized dict/object.
        """
        pass
    
    def decide(self):
        """
        Process observation and decide on action.
        Must complete within reasonable time.
        """
        pass
    
    def get_action(self) -> 'Action':
        """
        Return action to be executed.
        Must return Action object in standard format.
        """
        pass
    
    def learn(self, reward: float, done: bool, info: dict = None):
        """
        Optional: receive feedback (for future RL support).
        """
        pass
    
    def shutdown(self):
        """
        Cleanup on simulator shutdown.
        """
        pass
```

**CRITICAL CONSTRAINT:** Simulator has ZERO knowledge of brain implementation. Brain only receives observations and provides actions through defined interfaces.

---

### 2. OBSERVATION OBJECT (Standardized)

```python
# core/observation.py

@dataclass
class Observation:
    """
    Standardized observation from simulator.
    Only includes configured sensors.
    """
    
    timestamp: float                    # Simulation time (seconds)
    
    # Proprioception (always included)
    proprioception: dict = None
    # {
    #     "joint_positions": {...},     # Joint angles (radians)
    #     "joint_velocities": {...},    # Joint angular velocities
    #     "joint_accelerations": {...}, # Joint angular accelerations
    #     "body_position": [x, y, z],   # World position
    #     "body_velocity": [vx, vy, vz], # Linear velocity
    #     "body_rotation": [qx, qy, qz, qw], # Quaternion
    #     "body_angular_velocity": [...],
    # }
    
    # Vision (if enabled)
    vision: dict = None
    # {
    #     "rgb": numpy array (H, W, 3),
    #     "resolution": (height, width),
    #     "fov": degrees,
    # }
    
    # Depth (if enabled)
    depth: dict = None
    # {
    #     "depth_image": numpy array (H, W),
    #     "min_distance": float,
    #     "max_distance": float,
    # }
    
    # IMU (if enabled)
    imu: dict = None
    # {
    #     "acceleration": [x, y, z],
    #     "angular_velocity": [wx, wy, wz],
    #     "orientation": [qx, qy, qz, qw],
    # }
    
    # Touch/Contact (if enabled)
    touch: dict = None
    # {
    #     "contacts": [
    #         {"position": [...], "force": [...], "object_id": ...},
    #     ]
    # }
    
    # Metadata
    info: dict = None               # Additional info for debugging
    
    def __post_init__(self):
        """Validate observation consistency."""
        if self.timestamp is None:
            raise ValueError("Observation requires timestamp")
```

**KEY RULE:** Observation contains ONLY data from configured sensors. No hidden simulator state leaked.

---

### 3. ACTION OBJECT (Standardized & Validated)

```python
# core/action.py

@dataclass
class Action:
    """
    Standardized action from brain.
    Validated before physics execution.
    """
    
    joint_targets: dict = None      # {"joint_name": target_value}
                                    # For position control
    
    motor_commands: dict = None     # {"joint_name": force/torque}
                                    # For force control
    
    gripper_commands: dict = None   # {"gripper_name": open/close}
                                    # For future grippers
    
    forces: dict = None             # {"body_name": [fx, fy, fz]}
                                    # Direct force application
    
    torques: dict = None            # {"body_name": [tx, ty, tz]}
                                    # Direct torque application
    
    timestamp: float = None         # When action was generated
    
    def __post_init__(self):
        """Ensure at least one action specified."""
        if not any([self.joint_targets, self.motor_commands, 
                   self.gripper_commands, self.forces, self.torques]):
            raise ValueError("Action must specify at least one command")
```

---

### 4. PHYSICS ABSTRACTION (Interface Only)

```python
# physics/physics_engine.py

class PhysicsEngine:
    """
    Abstract physics engine.
    Encapsulates MuJoCo/Bullet/etc internally.
    AI never touches this directly - only simulator uses it.
    """
    
    def load_body(self, body_definition: dict):
        """Load body structure from config."""
        pass
    
    def reset(self, seed: int = None):
        """Reset physics to initial state."""
        pass
    
    def step(self, dt: float = 0.005):
        """Advance physics by timestep dt (seconds)."""
        pass
    
    def apply_action(self, action: Action):
        """
        Apply brain's action to physics.
        - Validate action
        - Convert to motor commands
        - Execute
        """
        pass
    
    def get_body_state(self) -> dict:
        """
        Return current body state (used by sensors).
        {
            "joint_positions": {...},
            "joint_velocities": {...},
            "body_position": [...],
            "body_velocity": [...],
            "body_rotation": [...],
            ...
        }
        """
        pass
    
    def get_contact_info(self) -> list:
        """Return list of current contacts."""
        pass
    
    def set_gravity(self, gravity: float):
        """Configure gravity."""
        pass
    
    def shutdown(self):
        """Cleanup."""
        pass
```

**IMPLEMENTATION:** MuJoCo backend or PyBullet backend - simulator doesn't care which.

---

### 5. BODY DEFINITION (YAML Configuration)

```yaml
# config/body_humanoid.yaml

body:
  type: humanoid
  mass: 70.0              # kg
  height: 1.75            # meters
  default_damping: 0.01

links:
  head:
    mass: 5.0
    inertia: [0.05, 0.05, 0.05]
    collision_shape: sphere
    radius: 0.1
    
  torso:
    mass: 20.0
    inertia: [0.5, 0.5, 0.3]
    collision_shape: box
    dimensions: [0.3, 0.5, 0.2]
    
  upper_arm_r:
    mass: 3.0
    inertia: [0.01, 0.01, 0.01]
    collision_shape: cylinder
    radius: 0.04
    length: 0.3
    
  # ... (continue for all body parts)

joints:
  neck:
    type: revolute
    parent: torso
    child: head
    axis: [0, 0, 1]              # Z-axis rotation
    range: [-45, 45]             # degrees
    max_torque: 20.0             # Nm
    damping: 0.1
    
  shoulder_r:
    type: revolute
    parent: torso
    child: upper_arm_r
    axis: [1, 0, 0]              # X-axis rotation
    range: [-120, 120]
    max_torque: 50.0
    damping: 0.1
    
  # ... (continue for all joints)

initial_state:
  joint_positions:
    neck: 0.0
    shoulder_r: 0.0
    # ... (all joints)
```

**KEY PRINCIPLE:** Zero hardcoded body structure. Everything loadable from config.

---

### 6. SENSOR SYSTEM (Modular & Extensible)

```python
# sensors/sensor_base.py

class Sensor:
    """Abstract sensor base class."""
    
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
    
    def observe(self, physics_state: dict) -> dict:
        """
        Return sensor reading.
        Input: current physics state
        Output: sensor observation dict
        """
        raise NotImplementedError
    
    def reset(self):
        """Reset sensor state (if any)."""
        pass

# sensors/proprioception.py

class ProprioceptionSensor(Sensor):
    """Joint positions, velocities, etc."""
    
    def observe(self, physics_state: dict) -> dict:
        return {
            "joint_positions": physics_state["joint_positions"],
            "joint_velocities": physics_state["joint_velocities"],
            "joint_accelerations": physics_state["joint_accelerations"],
            "body_position": physics_state["body_position"],
            "body_velocity": physics_state["body_velocity"],
            "body_rotation": physics_state["body_rotation"],
            "body_angular_velocity": physics_state["body_angular_velocity"],
        }

# sensors/imu.py

class IMUSensor(Sensor):
    """Accelerometer + gyroscope + orientation."""
    
    def observe(self, physics_state: dict) -> dict:
        return {
            "acceleration": calculate_acceleration(physics_state),
            "angular_velocity": physics_state["body_angular_velocity"],
            "orientation": physics_state["body_rotation"],
        }

# sensors/observation_builder.py

class ObservationBuilder:
    """Assemble final observation from configured sensors."""
    
    def __init__(self, sensors: dict):
        self.sensors = sensors
    
    def build(self, physics_state: dict, timestamp: float) -> Observation:
        """Create Observation object from enabled sensors."""
        obs_dict = {}
        
        for sensor_name, sensor in self.sensors.items():
            if sensor.enabled:
                obs_dict[sensor_name] = sensor.observe(physics_state)
        
        return Observation(
            timestamp=timestamp,
            proprioception=obs_dict.get('proprioception'),
            vision=obs_dict.get('vision'),
            depth=obs_dict.get('depth'),
            imu=obs_dict.get('imu'),
            touch=obs_dict.get('touch'),
            info={"sensors_enabled": list(obs_dict.keys())}
        )
```

---

### 7. ACTUATOR SYSTEM (Validated Commands)

```python
# actuators/actuator_base.py

class Actuator:
    """Abstract actuator."""
    
    def __init__(self, name: str, joint_name: str, config: dict):
        self.name = name
        self.joint_name = joint_name
        self.config = config
        
        # Limits
        self.max_torque = config.get('max_torque', 100.0)
        self.max_velocity = config.get('max_velocity', 100.0)
        self.max_force = config.get('max_force', 100.0)
        self.damping = config.get('damping', 0.01)
    
    def validate_command(self, value: float) -> (bool, float):
        """
        Validate and clamp command value.
        Returns: (is_valid, clamped_value)
        """
        if np.isnan(value) or np.isinf(value):
            return False, 0.0
        
        clamped = np.clip(value, -self.max_torque, self.max_torque)
        return True, clamped
    
    def execute(self, physics_engine, command: float):
        """Apply command to physics engine."""
        is_valid, clamped = self.validate_command(command)
        if not is_valid:
            raise ValueError(f"Invalid actuator command: {command}")
        return clamped

# actuators/motor.py

class MotorActuator(Actuator):
    """Joint motor (position or torque control)."""
    
    def __init__(self, name: str, joint_name: str, config: dict):
        super().__init__(name, joint_name, config)
        self.control_mode = config.get('control_mode', 'torque')
        # 'torque' or 'position'
    
    def execute(self, physics_engine, command: dict):
        """
        Execute motor command.
        command: {"target": value, "mode": "torque"/"position"}
        """
        is_valid, clamped = self.validate_command(command.get('target', 0.0))
        if not is_valid:
            raise ValueError(f"Invalid motor command: {command}")
        
        mode = command.get('mode', self.control_mode)
        physics_engine.set_joint_command(
            self.joint_name,
            clamped,
            mode
        )
```

---

### 8. ACTION VALIDATION (Safety Layer)

```python
# core/validator.py

class ActionValidator:
    """Validate all brain actions before physics execution."""
    
    def __init__(self, actuators: dict, config: dict):
        self.actuators = actuators
        self.config = config
        self.log = []
    
    def validate(self, action: Action) -> (bool, Action, list):
        """
        Validate action.
        Returns: (is_valid, validated_action, errors)
        """
        errors = []
        validated_action = Action()
        
        # Validate joint targets
        if action.joint_targets:
            for joint_name, target in action.joint_targets.items():
                if joint_name not in self.actuators:
                    errors.append(f"Unknown joint: {joint_name}")
                    continue
                
                if np.isnan(target) or np.isinf(target):
                    errors.append(f"Invalid value for {joint_name}: {target}")
                    continue
                
                is_valid, clamped = self.actuators[joint_name].validate_command(target)
                if not is_valid:
                    errors.append(f"Invalid command for {joint_name}: {target}")
                    continue
                
                validated_action.joint_targets[joint_name] = clamped
        
        # Similar validation for motor_commands, forces, torques...
        
        is_valid = len(errors) == 0
        if not is_valid and self.config.get('debug', False):
            self.log.append({"timestamp": time.time(), "errors": errors})
        
        return is_valid, validated_action, errors
```

---

### 9. MAIN SIMULATOR LOOP

```python
# simulator/simulator.py

class Simulator:
    """Main simulator class - orchestrates entire loop."""
    
    def __init__(self, config_path: str = "config/simulator_config.yaml"):
        # Load configs
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.physics = self._init_physics()
        self.body = self._init_body()
        self.sensors = self._init_sensors()
        self.actuators = self._init_actuators()
        self.validator = ActionValidator(self.actuators, self.config)
        self.observation_builder = ObservationBuilder(self.sensors)
        
        # Brain (will be injected externally)
        self.brain = None
        
        # Simulation state
        self.timestep = self.config.get('timestep', 0.005)
        self.current_time = 0.0
        self.step_count = 0
        self.running = False
        self.paused = False
    
    def set_brain(self, brain: BrainInterface):
        """Inject brain (dependency injection)."""
        self.brain = brain
    
    def reset(self, seed: int = None):
        """Reset entire simulation."""
        if seed is not None:
            np.random.seed(seed)
        
        self.physics.reset(seed)
        self.body.reset()
        self.sensors = self._init_sensors()
        self.current_time = 0.0
        self.step_count = 0
        
        if self.brain:
            self.brain.reset(context={"seed": seed})
    
    def step(self, n_steps: int = 1) -> Observation:
        """
        Execute simulation step.
        
        Complete loop:
        1. Advance physics
        2. Get physics state
        3. Build observation from sensors
        4. Send observation to brain
        5. Receive action from brain
        6. Validate action
        7. Apply action to actuators
        8. Return observation
        """
        observations = []
        
        for _ in range(n_steps):
            if self.paused:
                return observations[-1] if observations else None
            
            # 1. Advance physics
            self.physics.step(self.timestep)
            
            # 2. Get physics state
            physics_state = self.physics.get_body_state()
            
            # 3. Build observation from sensors
            observation = self.observation_builder.build(
                physics_state,
                self.current_time
            )
            observations.append(observation)
            
            # 4. Send observation to brain
            if self.brain:
                self.brain.observe(observation)
                
                # 5. Receive action from brain
                self.brain.decide()
                action = self.brain.get_action()
                
                # 6. Validate action
                is_valid, validated_action, errors = self.validator.validate(action)
                
                if not is_valid:
                    if self.config.get('debug', False):
                        print(f"Invalid action: {errors}")
                    # Use zero action if invalid
                    validated_action = Action(joint_targets={})
                
                # 7. Apply action to physics
                self.physics.apply_action(validated_action)
            
            # Update state
            self.current_time += self.timestep
            self.step_count += 1
        
        return observations[-1] if observations else None
    
    def pause(self):
        """Pause simulation."""
        self.paused = True
    
    def resume(self):
        """Resume simulation."""
        self.paused = False
    
    def shutdown(self):
        """Cleanup."""
        if self.brain:
            self.brain.shutdown()
        self.physics.shutdown()
    
    def get_state(self) -> dict:
        """Get full simulator state (for debugging)."""
        return {
            "current_time": self.current_time,
            "step_count": self.step_count,
            "physics_state": self.physics.get_body_state(),
            "paused": self.paused,
        }
    
    # Private methods for initialization...
    def _load_config(self, path: str) -> dict:
        """Load and validate config."""
        pass
    
    def _init_physics(self) -> PhysicsEngine:
        """Initialize physics engine."""
        pass
    
    def _init_body(self):
        """Load body from config."""
        pass
    
    def _init_sensors(self) -> dict:
        """Initialize sensors from config."""
        pass
    
    def _init_actuators(self) -> dict:
        """Initialize actuators from config."""
        pass
```

---

### 10. CONFIGURATION SYSTEM (YAML)

```yaml
# config/simulator_config.yaml

simulator:
  timestep: 0.005                    # 5ms = 200Hz
  max_timestep: 0.1
  real_time_factor: 1.0              # 1.0 = real-time, >1.0 = faster
  seed: 42
  debug: true
  headless: true

physics:
  engine: mujoco                      # mujoco or bullet
  gravity: [0, 0, -9.81]
  damping: 0.01
  friction: 0.5
  restitution: 0.0
  contact_tolerance: 0.001

body:
  config_path: config/body_humanoid.yaml
  initial_position: [0, 0, 0]
  initial_rotation: [0, 0, 0, 1]      # quaternion

sensors:
  proprioception:
    enabled: true
  vision:
    enabled: false                     # Phase 2
  depth:
    enabled: false                     # Phase 2
  imu:
    enabled: true
  touch:
    enabled: false                     # Phase 2

actuators:
  config_path: config/actuators_config.yaml

environment:
  floor_enabled: true
  floor_friction: 0.5
  floor_size: [10, 10]

logging:
  level: INFO
  file: logs/simulation.log
```

---

## SUCCESS CRITERIA (Phase 1)

Your implementation is complete and correct when:

### ✅ FUNCTIONAL REQUIREMENTS

1. **Brain API**
   - [x] Abstract interface defined
   - [x] No simulator logic in brain
   - [x] Can swap dummy brain → any other brain without code changes
   - [x] Brain receives observations, returns actions

2. **Humanoid Body**
   - [x] Loaded from YAML config
   - [x] 12+ DOF
   - [x] Physically simulated (not position-hacked)
   - [x] Can stand with gravity
   - [x] Joints have realistic limits

3. **Physics**
   - [x] Fixed timestep (0.005s)
   - [x] Gravity works correctly
   - [x] Collision detection on
   - [x] Deterministic (with seed)
   - [x] Runs 1000+ steps without crash

4. **Sensors**
   - [x] Proprioception returns current joint state
   - [x] IMU returns accelerations/orientations
   - [x] Only configured sensors included in observation
   - [x] No hidden info leaked

5. **Actuators**
   - [x] Motors control joints
   - [x] Commands validated (NaN, infinity, limits)
   - [x] Invalid actions logged/clamped
   - [x] Can execute 1000+ actions without error

6. **Observations & Actions**
   - [x] Observation object standardized
   - [x] Action object standardized
   - [x] All data serializable (JSON)
   - [x] Timestamps correct

7. **Configuration**
   - [x] Body fully configurable via YAML
   - [x] Zero hardcoded body parameters
   - [x] Physics params in config
   - [x] Sensor config in config
   - [x] Actuator config in config
   - [x] Config validation on startup

8. **Testing**
   - [x] Unit tests for each component
   - [x] Integration test (brain → physics → observation loop)
   - [x] Test with dummy brain (minimal implementation)
   - [x] >80% code coverage

### ✅ CODE QUALITY

- [x] All code documented (docstrings)
- [x] Type hints throughout
- [x] No simulator logic in brain code
- [x] No brain logic in simulator code
- [x] Clear separation of concerns
- [x] <5000 LOC for Phase 1
- [x] Production-ready (no TODOs in core code)

### ✅ DELIVERABLES

- [x] Working simulator
- [x] Example configurations (humanoid body)
- [x] Dummy brain implementation
- [x] Basic integration example
- [x] Comprehensive documentation
- [x] README with usage examples

---

## EXAMPLE USAGE (After Implementation)

```python
# examples/basic_simulation.py

from simulator import Simulator
from brain import DummyBrain

# Create simulator
sim = Simulator("config/simulator_config.yaml")

# Create and inject brain
brain = DummyBrain()
sim.set_brain(brain)

# Reset
sim.reset(seed=42)

# Run simulation
for episode in range(10):
    sim.reset()
    
    for step in range(1000):
        # Single step: physics → sensors → brain → action → physics
        observation = sim.step()
        
        print(f"Step {step}: {observation.proprioception['body_position']}")
        
        # Brain automatically runs in loop
        
    print(f"Episode {episode} complete")

sim.shutdown()
```

---

## IMPLEMENTATION PRIORITIES

**Do first:**
1. Config system (YAML loading)
2. Observation/Action objects
3. Physics abstraction + MuJoCo backend
4. Body loading from config
5. Sensor framework (proprioception only initially)
6. Actuator system + validation
7. Brain interface (abstract only)
8. Main loop
9. Tests
10. Documentation

**Do NOT do in Phase 1:**
- Rendering
- RL libraries
- Multi-agent
- Plugins
- Recording
- Real robot adapters
- Parallel environments

---

## CRITICAL REMINDERS

### 🚫 NEVER:
- Hardcode body structure
- Leak physics internals to brain
- Put brain logic in simulator
- Put simulator logic in brain
- Access physics directly from brain
- Assume specific brain type

### ✅ ALWAYS:
- Use abstract interfaces
- Validate all inputs
- Log invalid actions
- Use config files
- Test components independently
- Keep brain and physics separate

---

## RESOURCES & REFERENCES

**Physics Engines:**
- MuJoCo: https://mujoco.org/
- PyBullet: https://pybullet.org/

**Python Tools:**
- NumPy: https://numpy.org/
- PyYAML: https://pyyaml.org/

**Testing:**
- pytest: https://pytest.org/

**Python Standards:**
- Type hints: https://docs.python.org/3/library/typing.html
- Dataclasses: https://docs.python.org/3/library/dataclasses.html

---

## NEXT STEPS AFTER PHASE 1

Once Phase 1 is complete:

1. **Phase 2:** Add rendering/visualization
2. **Phase 3:** RL training framework
3. **Phase 4:** Multi-agent support
4. **Phase 5:** Real robot adapters
5. **Phase 6:** Recording/replay system
6. **Phase 7:** Parallel environments for training
7. **Phase 8:** Advanced sensors/actuators

**All without rewriting core simulator or brain API.**

---

## FINAL NOTE

This specification is designed for **one principle:**

> **THE AI BRAIN AND THE BODY ARE COMPLETELY SEPARATE.**

This means:
- Today: You can plug in a dummy brain
- Tomorrow: Plug in an LLM brain
- Next week: Plug in a RL agent brain
- Next month: Plug in NovaSocial's AI brain
- Future: Plug in anything without rewriting simulator

**Design for that flexibility from day one.**

---

**Good luck. Build it well. 🚀**
