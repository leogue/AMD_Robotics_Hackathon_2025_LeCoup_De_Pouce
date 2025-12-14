import subprocess
import time
import sys
import signal

# 🔧 TEMPS MAX PAR TÂCHE (en secondes)
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
    "--dataset.episode_time_s=10000", # On garde un temps long interne, c'est le script python qui coupera
    "--dataset.push_to_hub=False",
    "--dataset.num_episodes=1" # On vise 1 seul épisode
]

def sanitize_name(name):
    return name.replace(" ", "_").replace("-", "_")

def run_task_with_timeout(task_name, timeout):
    safe_task_name = sanitize_name(task_name)
    timestamp = int(time.time())
    unique_repo_id = f"lleeoogg/eval_LeCoup-De-Pouce_{safe_task_name}_{timestamp}"

    cmd = BASE_CMD_ARGS + [
        f"--dataset.repo_id={unique_repo_id}",
        f"--dataset.single_task={task_name}"
    ]

    print(f"\n{'='*60}")
    print(f"▶ DÉMARRAGE TÂCHE : {task_name}")
    print(f"⏱  Temps imparti  : {timeout} secondes")
    print(f"{'='*60}\n")

    # On lance le processus sans bloquer le script
    process = subprocess.Popen(cmd)

    try:
        # On attend soit la fin du process, soit le timeout
        process.wait(timeout=timeout)
        print(f"✅ Tâche '{task_name}' terminée naturellement (succès ou fin).")
    
    except subprocess.TimeoutExpired:
        # Si le temps est écoulé
        print(f"\n⏰ TEMPS ÉCOULÉ ({timeout}s) ! Arrêt forcé de la tâche...")
        process.send_signal(signal.SIGINT) # Envoie un Ctrl+C virtuel au robot
        
        try:
            process.wait(timeout=5) # On lui laisse 5s pour fermer proprement
        except subprocess.TimeoutExpired:
            process.kill() # Si il bloque encore, on tue le process
            print("💀 Processus tué brutalement.")
        
        print(f"⏹ Tâche '{task_name}' stoppée par le chrono.")

    except KeyboardInterrupt:
        # Si TU appuies sur Ctrl+C pendant le run
        print("\n✂️  Interruption manuelle détectée.")
        process.send_signal(signal.SIGINT)
        process.wait()
        # On relève l'erreur pour arrêter tout le script si c'est toi qui l'a demandé
        raise KeyboardInterrupt

# ---- MAIN ----

if __name__ == "__main__":
    try:
        # Tâche 1
        run_task_with_timeout("Pick up and give the glove", TASK_TIMEOUT)

 
        # Tâche 2
        run_task_with_timeout("Pick up and give the syringe", TASK_TIMEOUT)

    except KeyboardInterrupt:
        print("\n🛑 Arrêt total demandé par l'utilisateur.")
