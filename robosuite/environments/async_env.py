from __future__ import annotations
import logging
import mujoco
import multiprocessing as mp
import queue
import sys
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from functools import cached_property, partial
from multiprocessing.context import SpawnContext
from multiprocessing.queues import Queue as MPQueue
from typing import Any, Callable, Dict, Optional, Sequence, TYPE_CHECKING

import numpy as np
import robosuite as suite
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

if TYPE_CHECKING:
    import mujoco.viewer
    from robosuite.environments.base import MujocoEnv
    from robosuite.renderers.viewer import MjviewerRenderer
    from multiprocessing.context import SpawnProcess
    from multiprocessing.synchronize import Event


@dataclass
class Observation:
    """Observation produced by the asynchronous simulation."""
    data: OrderedDict
    time: float
    action: np.ndarray


@dataclass(frozen=True)
class Request:
    command: str
    payload: Any = None


@dataclass(frozen=True)
class Response:
    command: str
    payload: Any = None
    error: Optional[str] = None


class PeriodicEventGenerator:
    def __init__(self, period: float | int, start_time: float | int):
        assert period > 0, "Period must be strictly positive."
        self.period = period
        self.last_event_time = start_time

    def is_ready(self, timestamp: float | int) -> bool:
        return timestamp >= self.last_event_time + self.period

    def register_event(self, timestamp: float | int):
        self.last_event_time = timestamp


class RealTimeRateMeter:
    """Estimates the real-time rate of a simulation over wall-clock time windows."""

    def __init__(self, real_time: float, sim_time: float):
        self._anchor_real_time = real_time
        self._anchor_sim_time = sim_time
        self.rate: float = 1.0

    def update(self, real_time: float, sim_time: float):
        """Update RTR estimate based on elapsed time since last update."""
        real_elapsed = real_time - self._anchor_real_time
        sim_elapsed = sim_time - self._anchor_sim_time

        if real_elapsed > 0:
            self.rate = sim_elapsed / real_elapsed

        # Reset anchor timestamps at every update
        self._anchor_real_time = real_time
        self._anchor_sim_time = sim_time


class Queue (MPQueue):
    """
    Multiprocessing queue that supports drop-oldest publish and draining the latest item.
    """

    def publish(self, item: Any):
        """
        Put an item into the queue, dropping the oldest element if the queue is full.
        """
        while True:
            try:
                self.put_nowait(item)
                break
            except queue.Full:
                try:
                    self.get_nowait()
                except queue.Empty:
                    continue

    def drain(self) -> Any:
        """
        Drain all items from the queue and return the latest one.
        """
        latest: Optional[Any] = None
        while True:
            try:
                latest = self.get_nowait()
            except queue.Empty:
                break
        return latest


@dataclass
class SimulationWorkerConf:
    env_factory: Callable[[], MujocoEnv]
    control_freq: float
    observation_freq: float
    visualization_freq: Optional[float]
    real_time_rate_freq: float
    target_real_time_rate: float
    reward_freq: float
    start_event: Event
    stop_event: Event
    done_event: Event
    control_queue: Queue
    observation_queue: Queue
    request_queue: Queue
    reply_queue: Queue

class SimulationWorker:
    conf: SimulationWorkerConf
    env: MujocoEnv
    sync_peg: PeriodicEventGenerator
    obs_peg: PeriodicEventGenerator
    reward_peg: PeriodicEventGenerator
    viz_peg: Optional[PeriodicEventGenerator]
    rtr_peg: PeriodicEventGenerator
    rtr_meter: RealTimeRateMeter
    latest_action: np.ndarray
    is_action_new: bool
    anchor_sim_time: float
    anchor_real_time: float
    sim_step_counter: int

    def __init__(self, conf: SimulationWorkerConf):
        self.conf = conf

        # Instantiate environment in the worker process.
        self.env = self.conf.env_factory()

    @property
    def default_action(self) -> np.ndarray:
        return np.zeros_like(self.env.action_spec[0], dtype=np.float32)

    @property
    def sim_time(self) -> float:
        return float(self.env.sim.data.time)

    def real_time(self) -> float:
        return time.perf_counter()

    @property
    def observation_queue(self) -> Queue:
        return self.conf.observation_queue
    
    @property
    def control_queue(self) -> Queue:
        return self.conf.control_queue
    
    @property
    def request_queue(self) -> Queue:
        return self.conf.request_queue
    
    @property
    def reply_queue(self) -> Queue:
        return self.conf.reply_queue
    
    def reset(self):
        # Initialize time constants used for simulation
        self.env.initialize_time(self.conf.control_freq)

        # Reset latest action to default
        self.latest_action = self.default_action
        self.is_action_new = True

        # Drain queues
        self.observation_queue.drain()
        self.control_queue.drain()

        # Reset the MuJoCo environment
        self.env.reset()

        # Initialize sim step counter
        self.sim_step_counter = 0

        # Warmup: take one simulation step to trigger any first-run JIT compilation
        # (e.g., MuJoCo JIT on fresh installs)
        self.take_sim_step()

        # Initialize viewer: the first call can be slow.
        if self.conf.visualization_freq is not None:
            self.update_viewer()

        # Get simulation timestep
        assert self.env.model_timestep is not None
        assert self.env.model_timestep == self.env.sim.model.opt.timestep
        sim_timestep = self.env.model_timestep

        # Helper to snap frequency to nearest multiple of sim timestep and compute period in ticks
        def snap_frequency_to_sim_timestep(freq: float, name: str) -> int:
            period_seconds = 1.0 / freq
            period_ticks = round(period_seconds / sim_timestep)
            snapped_freq = 1.0 / (period_ticks * sim_timestep)
            if snapped_freq != freq:
                ROBOSUITE_DEFAULT_LOGGER.warning(
                    f"{name.capitalize()} frequency {freq:.2f} Hz is not a perfect multiple of "
                    f"simulation timestep ({sim_timestep}s). Snapping to {snapped_freq:.2f} Hz."
                )
            return period_ticks

        # Simulation tick PEG to publish observations
        self.obs_peg = PeriodicEventGenerator(
            period=snap_frequency_to_sim_timestep(self.conf.observation_freq, "observation"),
            start_time=self.sim_step_counter,
        )
        # Publish initial observation
        obs = self.get_observations()
        self.observation_queue.publish(obs)

        # Simulation tick PEG to publish rewards
        self.reward_peg = PeriodicEventGenerator(
            period=snap_frequency_to_sim_timestep(self.conf.reward_freq, "reward"),
            start_time=self.sim_step_counter,
        )

        # Simulation tick PEG to sync simulation time with real time
        self.sync_peg = PeriodicEventGenerator(
            period=snap_frequency_to_sim_timestep(self.conf.control_freq, "control"),
            start_time=self.sim_step_counter,
        )
        # Initialize time anchors for absolute time tracking (used by sync_time)
        # Simulation time and real time are synced at initialization.
        self.anchor_sim_time = self.sim_time
        self.anchor_real_time = self.real_time()

        # Wall-clock PEG for periodically invoking the RTR meter
        rtr_period = 1.0 / self.conf.real_time_rate_freq
        self.rtr_peg = PeriodicEventGenerator(
            period=rtr_period,
            start_time=self.anchor_real_time,
        )
        # Initialize RTR meter by creating it; computing RTR needs elapsed time anyway.
        self.rtr_meter = RealTimeRateMeter(self.anchor_real_time, self.anchor_sim_time)

        # Wall-clock PEG for visualization
        if self.conf.visualization_freq is not None:
            viz_period = 1.0 / self.conf.visualization_freq
            self.viz_peg = PeriodicEventGenerator(
                period=viz_period,
                start_time=self.anchor_real_time,
            )
            # Viewer has already been initialized; this was done before setting anchor timestamps
            # because the first call to update_viewer() can be slow.
        else:
            self.viz_peg = None

    def handle_request(self, request: Request):
        assert isinstance(request, Request), f"Expected Request, got {type(request)}"
        command = request.command

        if command == "reset":
            self.reset()
            self.reply_queue.put(Response(command=command, payload=None))
        elif command == "shutdown":
            self.conf.stop_event.set()
            self.reply_queue.put(Response(command=command, payload=None))
        elif command == "get_action_dim":
            self.reply_queue.put(Response(command=command, payload=self.env.action_dim))
        else:
            raise ValueError(f"Unknown command received by simulation worker: {request.command!r}")

    def compute_reward(self) -> float:
        action = self.latest_action.copy()
        reward, _, _ = self.env._post_action(action)
        return reward

    def get_observations(self) -> Observation:
        obs_data = self.env.viewer._get_observations() if self.env.viewer_get_obs else self.env._get_observations()
        return Observation(
            data=obs_data,
            time=self.sim_time,
            action=self.latest_action.copy(),
        )

    def take_sim_step(self):
        action = self.latest_action.copy()

        # The main MuJoCo step
        if self.env.lite_physics:
            self.env.sim.step1()
        else:
            self.env.sim.forward()
        self.env._pre_action(action, policy_step=self.is_action_new)
        if self.env.lite_physics:
            self.env.sim.step2()
        else:
            self.env.sim.step()
        self.env._update_observables()

        self.sim_step_counter += 1
        self.is_action_new = False

    def sync_time(self):
        # Calculate target real time based on anchor
        sim_elapsed = self.sim_time - self.anchor_sim_time
        real_elapsed = self.real_time() - self.anchor_real_time
        target_real_elapsed = sim_elapsed / self.conf.target_real_time_rate

        # Sleep until target time
        if target_real_elapsed > real_elapsed:
            time.sleep(target_real_elapsed - real_elapsed)

    def update_viewer(self):
        if not self.env.renderer:
            return

        if self.env.renderer == "mjviewer" and self.env.viewer is None:
            # need to launch again after it was destroyed
            self.env.initialize_renderer()

        if self.env.renderer == "mujoco" or self.env.renderer == "mjviewer":
            self.env.viewer.update()

    def update_viewer_overlays(self):
        observed_rtr = self.rtr_meter.rate
        self.env.viewer.viewer.set_texts([
            (None, mujoco.mjtGridPos.mjGRID_TOPLEFT, "Async. simulation time", f"{self.sim_time:.3f}s"),
            (None, mujoco.mjtGridPos.mjGRID_TOPRIGHT, "Real-time rate", f"{observed_rtr:.3f}")
        ])

    def main_loop(self):
        # Reset the environment
        self.reset()

        # Signal to the parent process that initialization is complete.
        self.conf.start_event.set()

        # Main worker loop
        # Note that the horizon is interpreted as seconds of simulation time
        while not self.conf.stop_event.is_set() and not self.conf.done_event.is_set():
            # Check for requests from the parent process.
            try:
                # Requests are blocking until there is a reply on thr reply queue.
                # So there will only be one request at a time.
                request = self.conf.request_queue.get_nowait()

                # If a request is succesfully received, handle it.
                self.handle_request(request)
            except queue.Empty:
                pass

            # Update the latest action from the control queue.
            if (drained_action := self.control_queue.drain()) is not None:
                self.latest_action = np.array(drained_action, dtype=np.float32)
                self.is_action_new = True

            # Take a step in the environment.
            self.take_sim_step()

            # Synchronize sim time with real time if either an observation needs to be produced now
            # or if its time now to periodically sync them.
            if self.obs_peg.is_ready(self.sim_step_counter) or self.sync_peg.is_ready(self.sim_step_counter):
                self.sync_time()
                # We register the sync event every time we call sync_time() even if it wasn't triggered
                # by the sync_peg. This resets the sync interval, but still maintains the guarantee that
                # control will be executed in simulation with a delay of no more than 1/control_freq.
                self.sync_peg.register_event(self.sim_step_counter)

            # Publish observation.
            if self.obs_peg.is_ready(self.sim_step_counter):
                observations = self.get_observations()
                self.observation_queue.publish(observations)
                self.obs_peg.register_event(self.sim_step_counter)

            # Publish reward.
            if self.reward_peg.is_ready(self.sim_step_counter):
                reward = self.compute_reward()
                self.reward_peg.register_event(self.sim_step_counter)

            # Get current real time.
            curr_real_time = self.real_time()

            # Update real-time rate meter.
            if self.viz_peg and self.rtr_peg.is_ready(curr_real_time):
                self.rtr_meter.update(curr_real_time, self.sim_time)
                self.rtr_peg.register_event(curr_real_time)

            # Update visualization.
            if self.viz_peg and self.viz_peg.is_ready(curr_real_time):
                self.update_viewer()
                self.update_viewer_overlays()
                self.viz_peg.register_event(curr_real_time)

            # Check if simulation time horizon has been reached.
            if self.sim_time >= self.env.horizon:
                self.conf.done_event.set()

        self.env.close()

    @staticmethod
    def runner(conf: SimulationWorkerConf):
        if sys.platform == "darwin" and conf.visualization_freq is not None:
            SimulationWorker._run_with_main_thread_viewer(conf)
        else:
            worker = SimulationWorker(conf)
            worker.main_loop()

    @staticmethod
    def _run_with_main_thread_viewer(conf: SimulationWorkerConf):
        """macOS subprocess viewer workaround.

        On macOS, GLFW/Cocoa requires that the viewer runs on the process's main
        thread.  We install a custom ``_MJPYTHON`` dispatcher so that when the
        simulation (running on a background thread) calls ``launch_passive``, the
        actual GLFW initialisation is forwarded to the main thread via a queue.
        """
        import queue as stdlib_queue
        from mujoco import viewer as mjviewer
        from mujoco.viewer import _MjPythonBase, _launch_internal

        dispatch_queue: stdlib_queue.Queue = stdlib_queue.Queue()

        class _SubprocessMjPython(_MjPythonBase):
            """Routes launch_passive viewer creation to the main thread."""
            def launch_on_ui_thread(self, model, data, handle_return, key_callback,
                                    show_left_ui=True, show_right_ui=True):
                dispatch_queue.put((model, data, handle_return, key_callback,
                                    show_left_ui, show_right_ui))

        # Install our dispatcher so launch_passive takes the macOS path.
        mjviewer._MJPYTHON = _SubprocessMjPython()

        worker = SimulationWorker(conf)
        sim_thread = threading.Thread(target=worker.main_loop, daemon=True)
        sim_thread.start()

        # Main thread: wait for the viewer-launch request from the simulation.
        launch_args = None
        while sim_thread.is_alive():
            try:
                launch_args = dispatch_queue.get(timeout=0.1)
                break
            except stdlib_queue.Empty:
                continue

        if launch_args is not None:
            model, data, handle_return, key_callback, show_left_ui, show_right_ui = launch_args

            # Clear _MJPYTHON so _launch_internal initialises GLFW normally.
            mjviewer._MJPYTHON = None

            # Run the GLFW event loop on the main thread (blocking) until
            # the viewer window is closed.
            _launch_internal(
                model, data,
                run_physics_thread=False,
                handle_return=handle_return,
                key_callback=key_callback,
                show_left_ui=show_left_ui,
                show_right_ui=show_right_ui,
            )

        # Viewer closed (or was never requested) — stop the simulation.
        conf.stop_event.set()
        sim_thread.join(timeout=10.0)


class ObservationStream:
    """
    Interface for consuming observations produced by the asynchronous simulation process.
    """

    def __init__(self, queue_obj: Queue, history: int = 1):
        if history <= 0:
            raise ValueError("ObservationStream history must be a positive integer.")
        self._queue = queue_obj
        self._history = deque(maxlen=history)
        self._lock = threading.Lock()

    def _drain(self, block: bool, timeout: Optional[float]) -> Optional[Observation]:
        first = True
        while True:
            try:
                if block and first:
                    if timeout is None:
                        item = self._queue.get(block=True)
                    else:
                        item = self._queue.get(block=True, timeout=timeout)
                else:
                    item = self._queue.get_nowait()
                with self._lock:
                    self._history.append(item)
                first = False
                block = False
            except queue.Empty:
                break
        with self._lock:
            return self._history[-1] if self._history else None

    def latest(self) -> Optional[Observation]:
        """
        Returns the latest observation without blocking.
        If the queue is empty, returns None.
        """
        return self._drain(block=False, timeout=None)

    def get(self, timeout: Optional[float] = None) -> Observation:
        """
        Returns the latest observation with blocking.
        If the queue is empty, waits for the observation to be available.
        """
        result = self._drain(block=True, timeout=timeout)
        if result is None:
            raise TimeoutError("Timed out while waiting for observation.")
        return result

    def snapshot(self) -> Sequence[Observation]:
        self._drain(block=False, timeout=None)
        with self._lock:
            return tuple(self._history)


class ControlStream:
    """
    Interface for pushing actions (controls) to the asynchronous simulation process.
    """

    def __init__(self, queue_obj: Queue):
        self._queue = queue_obj
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0

    def push(self, action: Sequence[float]):
        action_array = np.asarray(action, dtype=np.float32).copy()
        with self._lock:
            self._latest = action_array
            self._latest_timestamp = time.perf_counter()
        self._queue.publish(action_array)

    def latest(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def latest_timestamp(self) -> float:
        with self._lock:
            return self._latest_timestamp


class AsyncSimulation:
    """
    Runs a robosuite environment in a separate process at approximately real-time rate.
    The parent process (typically running the policy) interacts with the simulation through
    action and observation streams.

    The control frequency is defined as follows: the control period (= 1.0 / control_freq) is the
    maximum time interval between sending a new control command and when that control command will
    be executed in the simulation.
    • Functionally, it is the (minimum) rate at which the simulation time is synced with real time.
    • A higher control frequency means that the delay between sending a new control command and its
      execution in the simulation is smaller, but at the cost of more frequent synchronization with
      real time with smaller sleep intervals.

    Args:
        env_factory (Callable[[], MujocoEnv]): Factory function that creates a new environment instance.
        control_freq (float): The frequency of the control stream in Hz.
        observation_freq (float): The frequency of the observation stream in Hz.
        visualization_freq (Optional[float]): The frequency of the visualization stream in Hz.
        target_real_time_rate (float): The target real-time rate of the simulation.
        reward_freq (float): The frequency of the reward stream in Hz.
        history (int): The number of steps to store in the observation stream.
        ctx (Optional[SpawnContext]): The multiprocessing context to use.
    """

    def __init__(
        self,
        env_factory: Callable[[], MujocoEnv],
        control_freq: float = 50.0,
        observation_freq: float = 30.0,
        visualization_freq: Optional[float] = 30.0,
        real_time_rate_freq: float = 5.0,
        target_real_time_rate: float = 1.0,
        reward_freq: float = 10.0,
        history: int = 1,
        ctx: Optional[SpawnContext] = None,
    ):
        if control_freq <= 0:
            raise ValueError("control_freq must be > 0.")
        if observation_freq <= 0:
            raise ValueError("observation_freq must be > 0.")

        self._ctx = ctx or mp.get_context("spawn")
        self._start_event = self._ctx.Event()
        self._stop_event = self._ctx.Event()
        self._done_event = self._ctx.Event()
        self._control_queue = Queue(maxsize=8, ctx=self._ctx)
        self._observation_queue = Queue(maxsize=max(1, history), ctx=self._ctx)
        self._request_queue = Queue(maxsize=1, ctx=self._ctx)
        self._reply_queue = Queue(maxsize=1, ctx=self._ctx)
        self._process: Optional[SpawnProcess] = None

        self.control_stream = ControlStream(self._control_queue)
        self.observation_stream = ObservationStream(self._observation_queue, history=history)

        self.worker_conf = SimulationWorkerConf(
            env_factory,
            control_freq,
            observation_freq,
            visualization_freq,
            real_time_rate_freq,
            target_real_time_rate,
            reward_freq,
            self._start_event,
            self._stop_event,
            self._done_event,
            self._control_queue,
            self._observation_queue,
            self._request_queue,
            self._reply_queue
        )

    def start(self):
        if self._process and self._process.is_alive():
            raise RuntimeError("AsyncSimulation already running.")

        self._start_event.clear()
        self._stop_event.clear()
        self._done_event.clear()

        self._process = self._ctx.Process(
            target=SimulationWorker.runner,
            name="AsyncSimulationProcess",
            daemon=True,
            args=(self.worker_conf,),
        )
        self._process.start()

        # Wait for the simulation to start
        started = self._start_event.wait(timeout=10.0)
        if not started:
            self.stop(wait=False)
            raise TimeoutError("AsyncSimulation failed to start within 10 seconds.")

    def stop(self, wait: bool = True):
        if self._process is None:
            return
        if self.is_running():
            try:
                self._send_request("shutdown", timeout=2.0)
            except Exception:
                pass
        if not self._stop_event.is_set():
            self._stop_event.set()
        if wait and self._process.is_alive():
            self._process.join(timeout=10.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)
        self._process = None
        self._start_event.clear()

    def reset(self):
        if not self.is_running():
            raise RuntimeError("AsyncSimulation is not running.")
        self._send_request("reset", timeout=2.0)

    def _send_request(
        self,
        command: str,
        payload: Any = None,
        timeout: Optional[float] = None,
    ) -> Any:
        if not self.is_running():
            raise RuntimeError(f"AsyncSimulation is not running; cannot send {command!r} command.")

        request = Request(command=command, payload=payload)
        self._request_queue.put(request)
        try:
            response = self._reply_queue.get(block=True, timeout=timeout)
        except queue.Empty as exc:
            raise RuntimeError(f"Timed out waiting for response to command {command!r}.") from exc

        if not isinstance(response, Response):
            raise RuntimeError(
                f"Received unknown message type from asynchronous simulation: {response!r}"
            )

        if response.command != command:
            raise RuntimeError(
                f"Received mismatched response '{response.command}' for command '{command}'."
            )

        if response.error is not None:
            raise RuntimeError(
                f"Command '{command}' failed in asynchronous simulation: {response.error}"
            )

        return response.payload

    @cached_property
    def action_dim(self) -> int:
        if not self.is_running():
            raise RuntimeError("AsyncSimulation must be running to query action_dim.")
        result = self._send_request("get_action_dim", timeout=2.0)
        return int(result)

    def done(self) -> bool:
        return self._done_event.is_set()

    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def latest_observation(self) -> Optional[Observation]:
        return self.observation_stream.latest()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop(wait=True)


def make_async(
    env_name: str,
    *args,
    control_freq: float = 50.0,
    observation_freq: float = 30.0,
    viz_freq: Optional[float] = 30.0,
    real_time_rate_freq: float = 5.0,
    target_real_time_rate: float = 1.0,
    history: int = 1,
    **kwargs,
) -> AsyncSimulation:
    """
    Instantiates an asynchronous simulation of a robosuite environment.
    Args:
        env_name (str): Name of the robosuite environment to initialize
        *args: Additional arguments to pass to the specific environment class initializer
        control_freq (float): The frequency of the control stream in Hz.
        observation_freq (float): The frequency of the observation stream in Hz.
        viz_freq (Optional[float]): The frequency of the visualization stream in Hz.
        real_time_rate_freq (float): The frequency of the real-time rate updates in Hz.
        target_real_time_rate (float): The target real-time rate of the simulation.
        history (int): The number of steps to store in the observation stream.
        **kwargs: Additional arguments to pass to the specific environment class initializer
    Returns:
        AsyncSimulation: Asynchronous simulation of the robosuite environment.
    """
    return AsyncSimulation(
        env_factory = partial(
            suite.make,
            env_name,
            *args,
            control_freq = control_freq,
            **kwargs,
            ignore_done = True,  # always ignore done
        ),
        control_freq = control_freq,
        observation_freq = observation_freq,
        visualization_freq = viz_freq,
        real_time_rate_freq = real_time_rate_freq,
        target_real_time_rate = target_real_time_rate,
        history = history,
    )
