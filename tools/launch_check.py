"""Start the installed `pinnstudio` command and check that it keeps running
(used by the platform workflow after the one-command install)."""
import os
import subprocess
import sys
import time

exe = sys.argv[1]
env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
p = subprocess.Popen([exe], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
time.sleep(30)
alive = p.poll() is None
p.kill()
out = p.communicate()[0].decode("utf-8", "replace")
if not alive:
    msg = f"pinnstudio exited with code {p.returncode}:\n{out[-3000:]}"
    print(msg)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print("::error title=installed app did not stay open::" + msg.replace("%", "%25").replace("\n", "%0A"))
    sys.exit(1)
print("pinnstudio started and stayed open for 30 s")
