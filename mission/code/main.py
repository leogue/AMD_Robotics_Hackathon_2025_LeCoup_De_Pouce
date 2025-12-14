import subprocess
import time
import signal
import queue
import threading
import pyttsx3
from audio_detection import AudioKeywordDetector

# Maximum time per task (seconds)
TASK_TIMEOUT = 60

BASE_CMD_ARGS = [
    "lerobot-record",
    "--robot.type=so101_follower",
    "--robot.port=/dev/ttyACM1",
    "--robot.id=follower_arm",
    '--robot.cameras={camera1: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}',
    "--display_data=true",
    "--policy.path=lleeoogg/LeCoup-De-Pouce",
    "--policy.device=cuda",
    "--policy.empty_cameras=2",
    "--dataset.episode_time_s=10000",  # Long internal time, script will cut it
    "--dataset.push_to_hub=False",
    "--dataset.num_episodes=1"
]

# Task mapping: voice commands to task descriptions
VOICE_TASKS = {
    "glove": "Pick up and give the glove",
    "syringe": "Pick up and give the syringe",
    "pliers": "Pick up and give the pliers"
}

def sanitize_name(name):
    return name.replace(" ", "_").replace("-", "_")

class VoiceControlledRobot:
    """Voice-controlled robot task manager with interruption support."""

    def __init__(self):
        self.current_process = None
        self.current_task_name = None
        self.current_task_key = None  # Track which task is currently running
        self.command_queue = queue.Queue()
        self.detector = AudioKeywordDetector(verbose=True)
        self._running = True

        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        self.tts_lock = threading.Lock()  # Thread safety for TTS

        # Register voice commands with aliases from audio_detection.py
        self.detector.register_keyword("glove", lambda: self._on_task_command("glove"))
        self.detector.register_keyword("syringe", lambda: self._on_task_command("syringe"), aliases=["syrian", "surrender"])
        self.detector.register_keyword("pliers", lambda: self._on_task_command("pliers"), aliases=["player", "players", "playoffs"])
        self.detector.register_keyword("stop", lambda: self._on_stop_command(), aliases=["step"])

    def _speak(self, text):
        """
        Speak text using TTS engine (thread-safe).

        Args:
            text: Text to speak
        """
        with self.tts_lock:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()

    def _on_task_command(self, task_key):
        """Called when a task keyword is detected."""
        # Check if the same task is already running
        if self.current_task_key == task_key and self.current_process and self.current_process.poll() is None:
            print(f"\n[VOICE] '{task_key}' already running - ignoring duplicate command")
            return

        self.command_queue.put(("task", task_key))
        print(f"\n[VOICE] '{task_key}' command detected - queued")

        # Voice confirmation
        confirmation = f"{task_key.capitalize()} detected, starting task"
        threading.Thread(target=self._speak, args=(confirmation,), daemon=True).start()

    def _on_stop_command(self):
        """Called when stop keyword is detected."""
        self.command_queue.put(("stop", None))
        print(f"\n[VOICE] 'stop' command detected - stopping current task")

        # Voice confirmation
        confirmation = "Stop command received, stopping current task"
        threading.Thread(target=self._speak, args=(confirmation,), daemon=True).start()

    def _stop_current_task(self):
        """Stop the currently running task process."""
        if self.current_process and self.current_process.poll() is None:
            print(f"\n🛑 Stopping task '{self.current_task_name}'...")
            self.current_process.send_signal(signal.SIGINT)

            try:
                self.current_process.wait(timeout=5)
                print(f"✅ Task '{self.current_task_name}' stopped cleanly")
            except subprocess.TimeoutExpired:
                self.current_process.kill()
                print(f"💀 Task '{self.current_task_name}' killed forcefully")

            self.current_process = None
            self.current_task_name = None
            self.current_task_key = None

    def _start_task(self, task_key):
        """Start a new task."""
        if task_key not in VOICE_TASKS:
            print(f"❌ Unknown task: {task_key}")
            error_msg = f"Unknown task: {task_key}"
            threading.Thread(target=self._speak, args=(error_msg,), daemon=True).start()
            return

        task_name = VOICE_TASKS[task_key]
        safe_task_name = sanitize_name(task_name)
        timestamp = int(time.time())
        unique_repo_id = f"lleeoogg/eval_LeCoup-De-Pouce_{safe_task_name}_{timestamp}"

        cmd = BASE_CMD_ARGS + [
            f"--dataset.repo_id={unique_repo_id}",
            f"--dataset.single_task={task_name}"
        ]

        print(f"\n{'='*60}")
        print(f"▶ STARTING TASK: {task_name}")
        print(f"⏱  Max duration: {TASK_TIMEOUT} seconds")
        print(f"{'='*60}\n")

        self.current_process = subprocess.Popen(cmd)
        self.current_task_name = task_name
        self.current_task_key = task_key

    def _monitor_task(self):
        """Monitor current task for timeout."""
        if not self.current_process:
            return

        start_time = time.time()

        while self.current_process and self.current_process.poll() is None:
            # Check for timeout
            if time.time() - start_time > TASK_TIMEOUT:
                print(f"\n⏰ TIMEOUT ({TASK_TIMEOUT}s) reached!")
                timeout_msg = "Timeout reached"
                threading.Thread(target=self._speak, args=(timeout_msg,), daemon=True).start()
                self._stop_current_task()
                break

            # Check for new commands
            try:
                cmd_type, cmd_data = self.command_queue.get(timeout=0.1)

                if cmd_type == "stop":
                    self._stop_current_task()
                    break
                elif cmd_type == "task":
                    # New task requested - stop current and start new
                    self._stop_current_task()
                    self._start_task(cmd_data)
                    # Restart monitoring for new task
                    return self._monitor_task()

            except queue.Empty:
                continue

        # Task finished naturally
        if self.current_process and self.current_process.poll() is not None:
            print(f"\n✅ Task '{self.current_task_name}' completed naturally")
            completion_msg = "Task completed"
            threading.Thread(target=self._speak, args=(completion_msg,), daemon=True).start()
            self.current_process = None
            self.current_task_name = None
            self.current_task_key = None

    def run(self):
        """Main control loop."""
        print("\n" + "="*60)
        print("VOICE-CONTROLLED ROBOT SYSTEM")
        print("="*60)
        print("\nAvailable voice commands:")
        for key, description in VOICE_TASKS.items():
            print(f"  '{key}' -> {description}")
        print(f"  'stop' -> Stop current task")
        print("\nPress Ctrl+C to quit")
        print("="*60 + "\n")

        # Start audio detector
        self.detector.start(blocking=False)

        try:
            while self._running:
                # Wait for commands
                try:
                    cmd_type, cmd_data = self.command_queue.get(timeout=0.5)

                    if cmd_type == "task":
                        # Stop any running task and start new one
                        self._stop_current_task()
                        self._start_task(cmd_data)
                        self._monitor_task()
                    elif cmd_type == "stop":
                        self._stop_current_task()

                except queue.Empty:
                    continue

        except KeyboardInterrupt:
            print("\n\n✂️ Shutdown requested...")
        finally:
            self._stop_current_task()
            self.detector.stop()
            print("👋 Robot system stopped")

if __name__ == "__main__":
    robot = VoiceControlledRobot()
    robot.run()
