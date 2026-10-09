"""Serializable hyperparameters and strictly disjoint scenario seed pools."""

from dataclasses import asdict, dataclass, field
import math


@dataclass
class SACConfig:
    hidden_sizes: tuple = (256, 256)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    initial_alpha: float = 0.2
    automatic_entropy: bool = True
    target_entropy: float | None = None

    def validate(self):
        if not self.hidden_sizes or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        if not (0 <= self.gamma <= 1 and 0 < self.tau <= 1):
            raise ValueError("Invalid gamma/tau")
        if not all(math.isfinite(x) and x > 0 for x in (self.learning_rate, self.initial_alpha)):
            raise ValueError("Invalid learning_rate/alpha")
        if self.target_entropy is not None and not math.isfinite(self.target_entropy):
            raise ValueError("Invalid target_entropy")


@dataclass
class TrainingConfig:
    total_steps: int = 1_000_000
    max_episodes: int | None = None
    max_episode_steps: int = 1000
    physics_steps: int = 10
    arm_speed: float = math.radians(50)
    gripper_speed: float = 0.045
    replay_capacity: int = 1_000_000
    batch_size: int = 256
    start_steps: int = 10_000
    update_after: int = 1000
    updates_per_step: int = 1
    log_every: int = 5000
    evaluate_every: int = 10_000
    checkpoint_every: int = 25_000
    seed: int = 42
    train_seed_start: int = 42
    train_seed_stop: int = 43  # exclusive: one seed repeats the same scenario
    train_seeds: list | None = None  # Explicit seeds take precedence over the interval.
    validation_seed_start: int = 1_000_000
    validation_episodes: int = 20
    validation_seeds: list | None = None
    bc_critic_warmup_updates: int = 1000
    device: str = "cuda"
    torch_threads: int = 1
    xml_path: str | None = None
    sac: SACConfig = field(default_factory=SACConfig)

    def validate(self):
        self.sac.validate()
        for name in ("total_steps", "max_episode_steps", "physics_steps", "replay_capacity", "batch_size",
                     "updates_per_step", "log_every", "evaluate_every", "checkpoint_every",
                     "validation_episodes", "torch_threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("start_steps", "update_after", "seed", "train_seed_start", "validation_seed_start", "bc_critic_warmup_updates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.max_episodes is not None and (isinstance(self.max_episodes, bool) or not isinstance(self.max_episodes, int) or self.max_episodes < 1):
            raise ValueError("max_episodes must be positive")
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size exceeds replay_capacity")
        if not isinstance(self.train_seed_stop, int) or self.train_seed_start >= self.train_seed_stop:
            raise ValueError("Empty training seed pool")
        for name in ('train_seeds', 'validation_seeds'):
            seeds = getattr(self,name)
            if seeds is not None and (not isinstance(seeds,list) or not seeds or
                    any(isinstance(s,bool) or not isinstance(s,int) or s < 0 for s in seeds) or
                    len(set(seeds)) != len(seeds)):
                raise ValueError(f'{name} must be a nonempty list of distinct nonnegative integers')
        overlap = (any(self.is_validation_seed(s) for s in self.train_seeds) if self.train_seeds is not None
                   else any(self.is_training_seed(s) for s in self.validation_seeds) if self.validation_seeds is not None
                   else max(self.train_seed_start, self.validation_seed_start) < min(
                       self.train_seed_stop, self.validation_seed_start + self.validation_episodes))
        if overlap:
            raise ValueError("Training and validation seed pools overlap")
        if not all(math.isfinite(x) and x > 0 for x in (self.arm_speed, self.gripper_speed)):
            raise ValueError("Action speeds must be positive and finite")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["sac"] = SACConfig(**value.get("sac", {}))
        result = cls(**value)
        result.validate()
        return result

    def environment_kwargs(self):
        return dict(physics_steps=self.physics_steps, max_steps=self.max_episode_steps,
                    arm_speed=self.arm_speed, gripper_speed=self.gripper_speed, xml_path=self.xml_path)

    def is_training_seed(self,seed):
        return seed in self.train_seeds if self.train_seeds is not None else self.train_seed_start <= seed < self.train_seed_stop

    def is_validation_seed(self,seed):
        return seed in self.validation_seeds if self.validation_seeds is not None else self.validation_seed_start <= seed < self.validation_seed_start+self.validation_episodes

    def validation_seed_pool(self):
        return self.validation_seeds if self.validation_seeds is not None else range(self.validation_seed_start,self.validation_seed_start+self.validation_episodes)

    def sample_training_seed(self,rng):
        return int(rng.choice(self.train_seeds)) if self.train_seeds is not None else int(rng.integers(self.train_seed_start,self.train_seed_stop))


@dataclass
class BCConfig:
    epochs: int = 200
    batch_size: int = 256
    learning_rate: float = 3e-4
    validation_fraction: float = .2
    seed: int = 42
    patience: int = 30
    min_delta: float = 1e-6
    initial_std: float = .3
    evaluate_every: int = 20
    rollouts: bool = True
    device: str = 'auto'
    torch_threads: int = 1

    def validate(self):
        for name in ('epochs', 'batch_size', 'torch_threads'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('seed', 'patience', 'evaluate_every'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if not 0 < self.validation_fraction < 1:
            raise ValueError('validation_fraction must be between 0 and 1')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive and finite')
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError('min_delta must be nonnegative and finite')
        if not math.isfinite(self.initial_std) or not math.exp(-20) <= self.initial_std <= math.exp(2):
            raise ValueError('initial_std must be within the SAC log_std limits')
