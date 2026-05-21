#tensorboard --logdir=./ppo_logs/

import subprocess

result = subprocess.Popen(
    ["tensorboard", "--logdir=./ppo_logs/"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

for line in result.stdout:
    print(line, end="")  # already has newline

result.wait()