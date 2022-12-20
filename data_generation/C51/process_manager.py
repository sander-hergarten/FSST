import subprocess

agent_count = 3

procs = [
    subprocess.Popen(
        ["python", "hyperparameter_tuning.py"], shell=False, stdout=subprocess.PIPE
    )
]

sweep_id = procs[0].stdout.readline().decode("utf-8").split(": ")[-1]
print(sweep_id)

for _ in range(agent_count - 1):
    procs.append(
        subprocess.Popen(
            ["python", "hyperparameter_tuning.py", "--sweep_id", sweep_id],
            shell=False,
        )
    )
