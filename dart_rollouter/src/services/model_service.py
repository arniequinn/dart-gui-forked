"""
Model Service Pool - vLLM Service Management
"""

import asyncio
import os
import argparse
from typing import List, Dict, Any, Optional, Union
import subprocess
import aiohttp
import time
from contextlib import asynccontextmanager
import pynvml
import logging
import logging.handlers
import json
import base64
import torch

from fastapi import HTTPException, status
from pydantic import BaseModel, Field

import socket
from datetime import datetime

# Global flag: Whether NVML is available
NVML_AVAILABLE = False
try:
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except pynvml.NVMLError as e:
    logging.warning(f"NVML initialization failed, GPU monitoring functionality will be disabled: {e}")

def set_logger(log_file: str = "logs/model_service.log", log_level: int = logging.INFO):
    """
    Setup logger to output logs to both file and console
    """
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = f"logs/model_service_{timestamp}.log"
    # Create logger
    logger = logging.getLogger('model_service')
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        # Create file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=50*1024*1024, backupCount=100)  # 50MB per file, keep 5 backups
        file_handler.setLevel(log_level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Create formatter and add to handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    logger.info(f"Logger initialized with log file: {log_file}")
    return logger


# Create global logger instance
logger = set_logger()


# -------------------------------------------------------------------------
# Step 1: Define data models and core configuration
# -------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """
    Model service configuration class
    """
    ckpt_path: str = Field(..., description="Model checkpoint path, e.g. /data/models/llama-2-7b-chat-hf")
    base_port: int = Field(8000, description="Base port number used by vLLM service")
    replicas: int = Field(1, description="Expected number of service replicas")
    vllm_params: Dict[str, Any] = Field(default_factory=dict, description="Additional vLLM startup parameters")
    save_local: bool = Field(False, description="Whether to save locally")
    save_path: str = Field(default="./", description="Save path")
    
class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class GenerationRequest(BaseModel):
    messages: List[Dict[str, Any]]
    parameters: Optional[Dict[str, Any]] = None

class TokenizeRequest(BaseModel):
    prompt: str
    parameters: Optional[Dict] = None

class SaveRequest(BaseModel):
    messages: List[Dict]
    reward: float
    task_id: str
    trace_id: str
    
class ReloadRequest(BaseModel):
    new_ckpt_path: str = Field(..., description="New model checkpoint path")
    batch_size: int = Field(1, description="Rolling update batch size", gt=0)


# -------------------------------------------------------------------------
# Step 2: Simplify and refactor ModelServicePool class
# -------------------------------------------------------------------------

class ServiceInstance:
    """
    Use a data class to uniformly manage the state of each service instance.
    """
    def __init__(self, port: int, gpu_id: int, process: subprocess.Popen, ckpt_path: str):
        self.port = port
        self.gpu_id = gpu_id
        self.process = process
        self.endpoint = f"http://localhost:{port}"
        self.requests_in_flight = 0
        self.ckpt_path = ckpt_path

    def __repr__(self):
        return f"<ServiceInstance(port={self.port}, gpu_id={self.gpu_id}, pid={self.process.pid})>"


class GPUInstance:
    """
    Use a data class to uniformly manage the state of each GPU instance, including GPU-level NVML management
    """
    def __init__(self, gpu_id: int, gpu_memory_utilization: float = 0.9):
        self.gpu_id = gpu_id
        self.gpu_memory_utilization = gpu_memory_utilization
        self.is_available = False
        self._handle = None
        
        # Initialize GPU handle and check availability
        self._initialize_gpu()
        self.check_and_set_availability()

    def _initialize_gpu(self):
        """Initialize GPU handle, lazy load NVML"""
        global NVML_AVAILABLE
        if not NVML_AVAILABLE:
            logger.warning(f"GPU {self.gpu_id}: NVML unavailable, GPU monitoring disabled")
            self._handle = None
            return
            
        try:
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
        except pynvml.NVMLError as e:
            logger.error(f"Failed to initialize GPU {self.gpu_id}: {e}")
            self._handle = None

    def check_and_set_availability(self, gpu_memory_utilization: float = None) -> bool:
        """
        Check if GPU has sufficient memory and set is_available flag
        """
        if gpu_memory_utilization is None:
            gpu_memory_utilization = self.gpu_memory_utilization
            
        if self._handle is None:
            self.is_available = False
            return False
            
        try:
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            total_memory = memory_info.total
            free_memory = memory_info.free
            required_memory = int(total_memory * gpu_memory_utilization)
            self.is_available = free_memory >= required_memory
            return self.is_available
        except pynvml.NVMLError as e:
            logger.error(f"Error checking GPU {self.gpu_id} availability: {e}")
            self.is_available = False
            return False

    def __del__(self):
        """GPU instance destructor"""
        try:
            self._handle = None
        except Exception:
            pass

    def __repr__(self):
        return f"<GPUInstance(gpu_id={self.gpu_id}, is_available={self.is_available})>"
    
    def get_memory_info(self) -> dict:
        """Get current GPU memory information for debugging"""
        if self._handle is None:
            return {}
        try:
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return {
                'total': memory_info.total,
                'free': memory_info.free,
                'used': memory_info.used
            }
        except pynvml.NVMLError:
            return {}


class ModelServicePool:
    """
    1. GPU state management
    2. Service instance management
    """
    def __init__(self, model_cfg: ModelConfig):
        logger.info("Initializing ModelServicePool...")
        self.model_cfg = model_cfg
        self.default_ckpt_path = model_cfg.ckpt_path
        self.last_ckpt_path = model_cfg.ckpt_path
        self.base_port = model_cfg.base_port
        self.replicas = model_cfg.replicas
        self.vllm_params = model_cfg.vllm_params
        self.gpu_memory_utilization = model_cfg.vllm_params.get("gpu_memory_utilization", 0.9)
        self.save_local = model_cfg.save_local
        self.save_path = model_cfg.save_path
        
        # Optimized fine-grained locking mechanism
        self.instances_lock = asyncio.Lock()  # Protect service_instances dictionary
        self.gpu_lock = asyncio.Lock()        # Protect GPU_Instances dictionary
        self.config_lock = asyncio.Lock()     # Protect configuration changes
        
        # {gpu_id: ServiceInstance}
        self.service_instances: Dict[int, ServiceInstance] = {}
        # {gpu_id: GPUInstance}
        self.gpu_instances: Dict[int, GPUInstance] = {}
        
        self.monitoring_task: Optional[asyncio.Task] = None

    async def initialize(self):    
        """
        Initialize model service pool, including GPU instances and service instances.
        Used for initial startup
        """
        try:
            gpu_count = self._get_gpu_count()
            if gpu_count == 0:
                raise RuntimeError("No available GPUs found. Cannot start ModelServicePool.")
            
            self.replicas = min(self.model_cfg.replicas, gpu_count)
            if self.replicas < self.model_cfg.replicas:
                logger.warning(f"Warning: Requested replicas ({self.model_cfg.replicas}) > available GPU count ({gpu_count}). "
                      f"Setting replicas to {self.replicas}.")
            
            await self._init_gpu_instances()
            logger.info(await self.get_gpu_info())

            # Start initial services
            instances = await self._start_initial_services()

            logger.info("Waiting for model service pool to become ready...")
            if not await self.wait_for_model_pool_ready(instances):
                raise RuntimeError("Model service pool did not become ready within timeout.")
            else:
                logger.info("Model service pool is ready. Adding instances to pool...")
                async with self.instances_lock:
                    for instance in instances:
                        self.service_instances[instance.gpu_id] = instance

            # Start background monitoring task
            self.monitoring_task = asyncio.create_task(self._monitor_replicas())
            
            logger.info("ModelServicePool initialized and monitoring started.")
            return True

        except Exception as e:
            logger.error(f"Error during ModelServicePool initialization: {e}")
            await self.shutdown()
            return False

    async def shutdown(self):
        """Safely shutdown the entire model service pool."""
        logger.info("Final shutdown sequence initiated.")
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
        await self._shutdown_all_services()
        logger.info("ModelServicePool has been shut down.")

    async def _init_gpu_instances(self) -> List[GPUInstance]:
        """Initialize all GPU instances"""
        gpu_count = self._get_gpu_count()
        for gpu_id in range(gpu_count):
            self.gpu_instances[gpu_id] = GPUInstance(gpu_id, self.gpu_memory_utilization)

    async def _start_initial_services(self) -> List[ServiceInstance]:
        """Start the initial [replicas] number of service instances."""
        logger.info(f"Starting {self.replicas} initial model services...")
        
        # Use GPU lock to quickly get available GPUs
        async with self.gpu_lock:
            available_gpus = [
                instance.gpu_id for instance in self.gpu_instances.values() 
                if instance.is_available
            ]
        
        if not available_gpus:
            logger.warning("No available GPUs to start initial services.")
            return []

        gpus_to_use = available_gpus[:min(self.replicas, len(available_gpus))]
        
        # Asynchronously check port availability
        ports_to_use = await self._find_available_ports(len(gpus_to_use))
        
        if len(ports_to_use) < len(gpus_to_use):
            logger.warning(f"Warning: Only found {len(ports_to_use)} available ports for {len(gpus_to_use)} GPUs")
            gpus_to_use = gpus_to_use[:len(ports_to_use)]
        
        logger.info(f"Using GPUs: {gpus_to_use}, Ports: {ports_to_use}")
        
        # Start service instances (without holding lock)
        tasks = [
            self._add_new_service_instance(port, gpu_id, self.default_ckpt_path) 
            for port, gpu_id in zip(ports_to_use, gpus_to_use)
        ]
        instances = await asyncio.gather(*tasks)
        
        return instances

    async def add_service_by_id(self, gpu_id: int):
        """Add service instance by GPU ID"""
        ports = await self._find_available_ports(1)
        if not ports:
            return None
        
        port_to_use = ports[0]
        instance = await self._add_new_service_instance(port_to_use, gpu_id, self.default_ckpt_path)

        if not await self.wait_for_model_pool_ready([instance]):
            return None
        return instance

    async def _add_new_service_instance(self, port: int, gpu_id: int, ckpt_path: str) -> ServiceInstance:
        """Start a new service instance and add it to the pool."""
        logger.info(f"Attempting to start service on port {port} with GPU {gpu_id} from ckpt_path {ckpt_path}...")
        
        # Mark GPU as unavailable
        async with self.gpu_lock:
            if gpu_id in self.gpu_instances:
                self.gpu_instances[gpu_id].is_available = False
        
        proc = self._launch_vllm_process(port, gpu_id, ckpt_path)
        if not proc:
            logger.error(f"Failed to launch process on port {port}.")
            # Restore GPU availability
            async with self.gpu_lock:
                if gpu_id in self.gpu_instances:
                    self.gpu_instances[gpu_id].is_available = True
            raise RuntimeError(f"Failed to launch vLLM process on GPU {gpu_id}")
        
        instance = ServiceInstance(port, gpu_id, proc, ckpt_path)
        logger.info(f"Successfully started service instance: {instance}")
        
        # Start log monitoring
        asyncio.create_task(self._stream_process_output(proc, port))

        return instance

    def _launch_vllm_process(self, port: int, gpu_id: int, ckpt_path: str) -> Optional[subprocess.Popen]:
        """Launch vLLM subprocess."""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        vllm_command = [
            "vllm", "serve", ckpt_path,
            "--trust-remote-code",
            "--port", str(port)
        ]

        # Dynamically add parameters from vllm_params
        for key, value in self.vllm_params.items():
            cli_key = f"--{key}"
            
            if value == "store_true":
                vllm_command.append(cli_key)
            elif value is not None:
                # Otherwise, add key-value pairs
                vllm_command.append(cli_key)
                vllm_command.append(str(value))
                
        logger.info(f"Launching vLLM command: {vllm_command}")
        
        try:
            return subprocess.Popen(
                vllm_command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                env=env, 
                bufsize=1, 
                universal_newlines=True
            )
        except (OSError, FileNotFoundError) as e:
            logger.error(f"Error launching vLLM process on port {port}: {e}")
            return None

    async def _remove_service_instance(self, gpu_id: int, kill: bool = False):
        """Stop and clean up the specified vllm serve instance, and release GPU resources."""
        instance = None
        async with self.instances_lock:
            instance = self.service_instances.pop(gpu_id, None)

        if not instance:
            logger.warning(f"No service instance found on GPU {gpu_id} to remove.")
            return

        logger.info(f"Shutting down service on GPU {gpu_id}, port {instance.port} (PID: {instance.process.pid if instance.process else 'N/A'})...")
        
        if instance.process and instance.process.poll() is None:
            proc = instance.process
            try:
                # First try graceful termination
                if kill:
                    proc.kill()
                else:
                    proc.terminate()
                
                # Wait for process termination, increase timeout
                await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=10)
                
            except asyncio.TimeoutError:
                # Force kill process
                logger.warning(f"Process {proc.pid} did not terminate gracefully. Force killing...")
                try:
                    proc.kill()
                    await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=5)
                except (asyncio.TimeoutError, ProcessLookupError):
                    logger.warning(f"Process {proc.pid} already terminated or inaccessible")
            
            # Ensure process group is also cleaned up
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except (ProcessLookupError, OSError, PermissionError):
                pass
        
        # Release GPU resources
        async with self.gpu_lock:
            if gpu_id in self.gpu_instances:
                self.gpu_instances[gpu_id].is_available = True
                logger.info(f"GPU {gpu_id} has been marked as available again")
        
        logger.info(f"Service on GPU {gpu_id} has been completely removed and GPU released")

    async def _shutdown_all_services(self):
        """Shutdown all running service instances."""
        logger.info("Shutting down all model services...")
        
        # Get all GPU IDs
        async with self.instances_lock:
            gpu_ids = list(self.service_instances.keys())
        
        shutdown_tasks = [self._remove_service_instance(gpu_id) for gpu_id in gpu_ids]
        await asyncio.gather(*shutdown_tasks)
        
        async with self.instances_lock:
            self.service_instances.clear()

    async def _shutdown_n_services(self, gpu_ids: List[int]):
        """
        Stop and remove the specified number of model service instances.
        """
        logger.info(f"Attempting to remove services on GPUs: {gpu_ids}")
        if not gpu_ids:
            return []

        shutdown_tasks = [self._remove_service_instance(gpu_id) for gpu_id in gpu_ids]
        await asyncio.gather(*shutdown_tasks)
        return gpu_ids
    
    async def _add_n_service_instance(self, count: int):
        """
        Start the specified number of new model service instances.
        """
        logger.info(f"Attempting to add {count} new model service(s)...")
        if count <= 0:
            return []

        # Quickly check available GPUs (hold lock for very short time)
        async with self.gpu_lock:
            available_gpus = [
                gpu.gpu_id for gpu in self.gpu_instances.values()
                if gpu.check_and_set_availability()
            ]

        if not available_gpus:
            logger.warning("Warning: No available GPUs to start new services.")
            return []

        gpus_to_use = available_gpus[:min(count, len(available_gpus))]
        logger.info(f">>>>>>> gpu to run {gpus_to_use}")
        
        if len(gpus_to_use) < count:
            logger.warning(f"Warning: Not enough free GPUs. Will start {len(gpus_to_use)} instead of {count}.")

        # Asynchronously check port availability (without holding lock)
        ports_to_use = await self._find_available_ports(len(gpus_to_use))
        
        if len(ports_to_use) < len(gpus_to_use):
            logger.warning(f"Warning: Not enough free ports. Starting {len(ports_to_use)} services.")
            gpus_to_use = gpus_to_use[:len(ports_to_use)]

        if not gpus_to_use:
            return []

        logger.info(f"Starting services on GPUs: {gpus_to_use}, Ports: {ports_to_use}")
        
        # Start service instances (without holding lock)
        tasks = [
            self._add_new_service_instance(port, gpu_id, self.default_ckpt_path) 
            for port, gpu_id in zip(ports_to_use, gpus_to_use)
        ]
        instances = await asyncio.gather(*tasks)
        
        # Wait for services to be ready (without holding lock)
        if not await self.wait_for_model_pool_ready(instances):
            raise Exception("add new instance failed")
        
        # Quickly add instances to dictionary
        async with self.instances_lock:
            for instance in instances:
                self.service_instances[instance.gpu_id] = instance
        
        return True

    async def _monitor_replicas(self):
        """Background monitoring task, regularly checks and maintains replica count."""
        await asyncio.sleep(120)
        while True:
            try:
                await self._check_all_service_health()
                await self._ensure_replicas()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring task: {e}")
            await asyncio.sleep(30)

    async def _ensure_replicas(self):
        """Core recovery logic: Check current service status, start missing replicas."""
        # Quickly get current status
        async with self.instances_lock:
            current_service_gpus = set(self.service_instances.keys())
            num_active = len(current_service_gpus)

        if num_active >= self.replicas:
            return

        logger.info(f"Replica check: Found {num_active}/{self.replicas} active instances. Attempting to restore...")
        needed = self.replicas - num_active
        
        # Quickly get available GPUs and ports
        async with self.gpu_lock:
            available_gpus = [
                instance.gpu_id for instance in self.gpu_instances.values() 
                if instance.check_and_set_availability()
            ]
        
        # Asynchronously check ports
        ports_to_create = await self._find_available_ports(min(len(available_gpus), needed))
        
        if not available_gpus:
            logger.warning("Warning: Cannot restore replicas, no free GPUs available.")
            return

        num_to_start = min(len(ports_to_create), len(available_gpus), needed)
        
        if num_to_start > 0:
            logger.info(f"Found resources to start {num_to_start} new instance(s).")
            
            # Start instances (without holding lock)
            tasks = [
                self._add_new_service_instance(
                    ports_to_create[i], 
                    available_gpus[i], 
                    self.default_ckpt_path
                ) for i in range(num_to_start)
            ]
            instances = await asyncio.gather(*tasks)
            
            # Wait for readiness
            if not await self.wait_for_model_pool_ready(instances):
                raise Exception("add new instance failed")
            
            # Quickly add to dictionary
            async with self.instances_lock:
                for instance in instances:
                    self.service_instances[instance.gpu_id] = instance

    async def _check_all_service_health(self):
        """
        Asynchronously check the health status of all service instances.
        If unhealthy instances are found, try to shut down and remove them.
        """
        logger.info("Checking health of all running services...")
        
        # Get instance snapshot (hold lock for very short time)
        async with self.instances_lock:
            current_instances = list(self.service_instances.values())
        
        if not current_instances:
            return
        
        health_checks = [
            self._check_service_health(instance.port) 
            for instance in current_instances
        ]
        results = await asyncio.gather(*health_checks, return_exceptions=True)
        
        # Handle unhealthy instances (without holding lock)
        unhealthy_instances = []
        for instance, is_healthy in zip(current_instances, results):
            if isinstance(is_healthy, Exception) or not is_healthy:
                logger.warning(f"Service on GPU {instance.gpu_id} (port {instance.port}) is unhealthy. Attempting to remove.")
                unhealthy_instances.append(instance.gpu_id)
            else:
                logger.info(f"Service on GPU {instance.gpu_id} (port {instance.port}) is healthy.")
        
        # Asynchronously remove unhealthy instances
        if unhealthy_instances:
            removal_tasks = [
                self._remove_service_instance(gpu_id) 
                for gpu_id in unhealthy_instances
            ]
            await asyncio.gather(*removal_tasks)

    async def _check_service_health(self, port: int, timeout: int = 5) -> bool:
        """Check the health of a single service through vLLM's /health endpoint."""
        url = f"http://localhost:{port}/health"
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(url) as response:
                    return response.status == 200
        except Exception:
            return False

    async def _wait_for_service_ready(self, instance: ServiceInstance, timeout: int = 1200) -> bool:
        """Wait for the service on the specified GPU to be ready."""
        start_time = time.time()
        logger.info(f"Waiting for service on GPU {instance.gpu_id} to be ready...")
        
        while time.time() - start_time < timeout:
            if not instance.process or instance.process.poll() is not None:
                logger.error(f"Process for GPU {instance.gpu_id} exited prematurely.")
                return False

            if await self._check_service_health(instance.port):
                logger.info(f"Service on GPU {instance.gpu_id} (port {instance.port}) is ready.")
                return True
            
            # Yield control to avoid blocking event loop
            await asyncio.sleep(1)
        
        logger.error(f"Timeout: Service on GPU {instance.gpu_id} did not become ready within {timeout}s.")
        return False

    async def wait_for_model_pool_ready(self, instances: List[ServiceInstance], timeout: int = 600):
        """
        Wait for at least one model service instance to start and be healthy, until the expected replica count is reached.
        """
        if not instances:
            return False
            
        ports = [inst.port for inst in instances]
        
        logger.info(f"Waiting for {len(ports)} service(s) to become ready (timeout: {timeout}s)...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            num_ready = 0
            tasks = [self._check_service_health(port) for port in ports]
            results = await asyncio.gather(*tasks)
            
            for result in results:
                if isinstance(result, bool) and result:
                    num_ready += 1
            
            if num_ready >= len(ports):
                logger.info(f"All {len(ports)} service(s) are ready.")
                return True
            
            # Yield control to avoid blocking event loop
            await asyncio.sleep(1)
        
        logger.error(f"Timeout: {len(instances)} service(s) did not become ready within {timeout}s.")
        return False

    def _get_gpu_count(self) -> int:
        """Get available GPU count."""
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            devices = os.environ["CUDA_VISIBLE_DEVICES"]
            if devices:
                return len(devices.split(","))
        try:
            result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                return len([line for line in lines if line.startswith('GPU ')])
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        return 0

    def _get_available_gpus(self) -> List[int]:
        """Get list of all GPU IDs."""
        return list(range(self._get_gpu_count()))

    async def _stream_process_output(self, proc: subprocess.Popen, port: int, max_log_length: int = 400):
        """Asynchronously read and print subprocess output stream, truncate overly long content."""
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

        async def reader(stream, prefix):
            while True:
                line = await asyncio.to_thread(stream.readline)
                if not line:
                    break

                clean_line = ansi_escape.sub('', line.rstrip())

                # Only keep first max_log_length characters
                if len(clean_line) > max_log_length:
                    clean_line = f"{clean_line[:max_log_length]}... (truncated {len(clean_line) - max_log_length} chars)"

                logger.info(f"[{prefix}] {clean_line}")

        stdout_task = asyncio.create_task(reader(proc.stdout, f"Port {port} STDOUT"))
        stderr_task = asyncio.create_task(reader(proc.stderr, f"Port {port} STDERR"))

        return_code = await asyncio.to_thread(proc.wait)
        logger.info(f"Process for port {port} terminated with return code {return_code}.")

        stdout_task.cancel()
        stderr_task.cancel()
        
    
    @asynccontextmanager
    async def _get_endpoint_for_request(self):
        """Use async context manager to gracefully handle endpoint acquisition and counter management."""
        instance = None
        try:
            async with self.instances_lock:
                active_instances = list(self.service_instances.values())
                if not active_instances:
                    raise Exception("No available endpoints in the pool.")
                
                instance = min(active_instances, key=lambda x: x.requests_in_flight)
                instance.requests_in_flight += 1
            
            logger.info(f"Routing request to {instance.endpoint} (GPU {instance.gpu_id}, in-flight: {instance.requests_in_flight})")
            yield instance
        
        finally:
            if instance:
                async with self.instances_lock:
                    instance.requests_in_flight -= 1

    async def generate(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Use load balancing strategy to send a chat request to the model service pool."""
        try:
            async with self._get_endpoint_for_request() as instance:
                logger.info(f"Using Service Instance ->>> {instance}")
                if not instance:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                        detail="Model service pool is not ready or has no available endpoints."
                    )
                
                url = f"{instance.endpoint}/v1/chat/completions"
                data = {"model": instance.ckpt_path, "messages": messages, **kwargs}

                if self.save_local:
                    try:
                        last_image_item = messages[-1]["content"][-1]
                        assert last_image_item.get("type") == "image_url"
                        last_image_url = last_image_item.get("image_url", {}).get("url", "")
                        assert last_image_url.startswith("data:image")
                        header, encoded = last_image_url.split(",", 1)

                        task_id = kwargs.get("task_id")
                        trace_id = kwargs.get("trace_id")
                        step = kwargs.get("step")
                        save_dir = os.path.join(self.save_path, f"{task_id}_{trace_id}")
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"image_{int(step)}.png")
                        with open(save_path, "wb") as f:
                            f.write(base64.b64decode(encoded))
                    except Exception as e:
                        logger.error(f"❌ Failed to decode or save image: {e}")
                        raise

                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=data) as response:
                        if not response.ok:
                            error_text = await response.text()
                            raise HTTPException(status_code=response.status, detail=f"API call failed: {error_text}")
                        
                        response_data = await response.json()
                        
                        if self.save_local:
                            try:
                                content = response_data["choices"][0]["message"]["content"]
                                model = response_data["model"]

                                logp_list, token_id_list = None, None

                                # If logprobs requested in parameters
                                if kwargs.get("logprobs", False):
                                    try:
                                        logp_list = [
                                            item["logprob"]
                                            for item in response_data["choices"][0]["logprobs"]["content"]
                                        ]
                                    except (KeyError, IndexError, TypeError):
                                        logp_list = None

                                # If return token_id requested in parameters
                                if kwargs.get("return_tokens_as_token_ids", False):
                                    try:
                                        token_id_list = [
                                            int(item["token"].split("token_id:")[1])
                                            for item in response_data["choices"][0]["logprobs"]["content"]
                                            if "token_id:" in item["token"]
                                        ]
                                    except (KeyError, IndexError, TypeError, ValueError):
                                        token_id_list = None
                                
                                task_id = kwargs.get("task_id")
                                trace_id = kwargs.get("trace_id")
                                step = kwargs.get("step")
                                save_dir = os.path.join(self.save_path, f"{task_id}_{trace_id}")
                                os.makedirs(save_dir, exist_ok=True)
                                save_path = os.path.join(save_dir, f"data_for_step_{int(step)+1}.pt")

                                data_to_save = {
                                    "logp": torch.tensor(logp_list).cpu() if logp_list is not None else torch.tensor([]).cpu(),
                                    "token_ids": torch.tensor(token_id_list).cpu() if token_id_list is not None else torch.tensor([]).cpu(),
                                }

                                torch.save(data_to_save, save_path)
                            except Exception as e:
                                logger.error(f"❌ Failed to save logp/token_id tensors: {e}")
                                raise

                        try:
                            return response_data
                        except (KeyError, IndexError, TypeError) as e:
                            raise HTTPException(
                                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                                detail=f"Failed to parse response: {e}. Full response: {response_data}"
                            )

        except Exception as e:
            logger.error(f"Error during generate call: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    async def save(self, messages: List[Dict], reward: float, task_id: str, trace_id: str):
        if not self.save_local:
            return {"status": "skipped"}

        try:
            save_dir = os.path.join(self.save_path, f"{task_id}_{trace_id}")
            os.makedirs(save_dir, exist_ok=True)

            # Save messages
            messages_path = os.path.join(save_dir, f"final_messages.json")
            with open(messages_path, "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False, indent=2)

            # Save reward
            reward_path = os.path.join(save_dir, f"reward.txt")
            with open(reward_path, "w") as f:
                f.write(str(reward))

            return {"status": "success"}

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save data: {e}"
            )
        
    async def tokenize(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """Call the /v1/tokenize endpoint of the model service pool"""
        try:
            async with self._get_endpoint_for_request() as instance:
                logger.info(f"Using Service Instance for tokenize ->>> {instance}")
                if not instance:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Model service pool is not ready or has no available endpoints."
                    )

                url = f"{instance.endpoint}/tokenize"
                data = {"model": instance.ckpt_path, "prompt": input_text}

                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=data) as response:
                        if not response.ok:
                            error_text = await response.text()
                            raise HTTPException(status_code=response.status, detail=f"Tokenize API failed: {error_text}")
                        
                        response_data = await response.json()
                        try:
                            return response_data
                        except (KeyError, IndexError, TypeError) as e:
                            raise HTTPException(
                                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Failed to parse tokenize response: {e}. Full response: {response_data}"
                            )
        except Exception as e:
            logger.error(f"Error during tokenize call: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    async def _find_available_ports(self, count: int, start_port: int = None, continuous: bool = False) -> List[int]:
        """
        Asynchronously discover the specified number of available ports.
        """
        if start_port is None:
            start_port = self.base_port
            
        available_ports = []
        port = start_port
        max_port = start_port + 1000  # Prevent infinite loop
        
        def _check_port_bind(port_num: int) -> bool:
            """Synchronously check if port is available"""
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("0.0.0.0", port_num))
                    return True
            except OSError:
                return False
        
        while len(available_ports) < count and port <= max_port:
            is_free = await asyncio.to_thread(_check_port_bind, port)
            
            if is_free:
                if continuous and available_ports and port != available_ports[-1] + 1:
                    available_ports.clear()
                    available_ports.append(port)
                else:
                    available_ports.append(port)
                
                if len(available_ports) == count:
                    break
            else:
                if continuous and available_ports:
                    available_ports.clear()
            
            port += 1
            
        if len(available_ports) < count:
            logger.warning(f"Warning: Could only find {len(available_ports)}/{count} available ports")
            
        return available_ports[:count]

    async def reload(self, new_ckpt_path: str):
        """
        Full delete and restart mode, temporarily not providing fastapi interface
        """
        logger.info(f"\n--- Reloading model pool to: {new_ckpt_path} ---")
        
        # Update configuration
        async with self.config_lock:
            self.default_ckpt_path = new_ckpt_path
        
        # Pause background monitoring
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
        
        # Shutdown all services
        await self._shutdown_all_services()
        
        # Re-initialize
        await self.initialize()
        
        logger.info("--- Model pool reloaded successfully ---")
        return True

    async def roll_reload(self, new_ckpt_path: str, batch_size: int = 1):
        """
        Smoothly reload the model through rolling updates.
        Update [batch_size] instances at a time.
        """

        if self.replicas >= 4:
            batch_size = max(2, batch_size)
        if self.replicas < 4:
            batch_size = 1

        logger.info(f"\n--- Rolling reload to new model: {new_ckpt_path} (batch size: {batch_size}) ---")
        
        # Pause background monitoring to prevent interference with update process
        if self.monitoring_task and not self.monitoring_task.done():
            logger.info("pause monitoring task")
            self.monitoring_task.cancel()

        # Quickly get old instance information (hold lock for very short time)
        async with self.instances_lock:
            old_instances_gpu_ids = [
                inst.gpu_id for inst in self.service_instances.values()
                if inst.ckpt_path != new_ckpt_path
            ]
        
        logger.info(f">>>>> old gpu ids : {old_instances_gpu_ids}")
        
        if not old_instances_gpu_ids:
            logger.info("All instances are already running the target model. Reload skipped.")
            # Restart monitoring
            self.monitoring_task = asyncio.create_task(self._monitor_replicas())
            return True

        # Set default model path to new path
        async with self.config_lock:
            self.last_ckpt_path = self.default_ckpt_path
            self.default_ckpt_path = new_ckpt_path

        # Update in batches, yield control between batches
        for i in range(0, len(old_instances_gpu_ids), batch_size):
            batch_gpu_ids = old_instances_gpu_ids[i:i + batch_size]
            logger.info(f"--- Reloading batch: GPUs {batch_gpu_ids} ---")

            # 1. Remove old instances
            logger.info(f"Removing {len(batch_gpu_ids)} old instance(s)...")
            await self._shutdown_n_services(batch_gpu_ids)
            
            # 2. Start new instances
            logger.info(f"Adding {len(batch_gpu_ids)} new instance(s) with new model...")
            await self._add_n_service_instance(len(batch_gpu_ids))
            
            # Yield control to allow other coroutines to execute
            await asyncio.sleep(0)

        # Restart background monitoring
        logger.info("restart monitoring task")
        self.monitoring_task = asyncio.create_task(self._monitor_replicas())
        logger.info(f"--- Rolling reload finished with new model: {new_ckpt_path} ---")
        return True

    async def get_endpoints(self) -> List[str]:
        """
        Get all vllm serve endpoints internally
        Old version ModelServicePool method, will be discarded if not needed
        """
        async with self.instances_lock:
            return [instance.endpoint for instance in self.service_instances.values()]

    async def get_status(self) -> List[Dict]:
        """
        Get status of all service instances
        """
        async with self.instances_lock:
            return [
                {
                    "gpu_id": inst.gpu_id,
                    "port": inst.port,
                    "endpoint": inst.endpoint,
                    "ckpt_path": inst.ckpt_path,
                    "requests_in_flight": inst.requests_in_flight,
                    "pid": inst.process.pid if inst.process else None
                }
                for inst in self.service_instances.values()
            ]
    
    async def get_gpu_info(self) -> List[Dict]:
        """Get GPU status"""
        async with self.gpu_lock:
            return [
                {
                    "gpu_id": ginst.gpu_id,
                    "is_available": ginst.is_available
                }
                for ginst in self.gpu_instances.values()
            ]
    
    async def get_checkpoint_info(self) -> Dict[str, Optional[str]]:
        """Get checkpoint path information"""
        async with self.config_lock:
            return {
                "current_ckpt_path": self.default_ckpt_path,
                "last_ckpt_path": self.last_ckpt_path
            }
