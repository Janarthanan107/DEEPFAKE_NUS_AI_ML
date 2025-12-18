import os
import json
import subprocess

def try_auth(username, key):
    print(f"Testing {username}...")
    config = {"username": username, "key": key}
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
        json.dump(config, f)
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
    
    cmd = "kaggle competitions list"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ SUCCESS with {username}!")
        return True
    else:
        print(f"❌ Failed with {username}. Error: {result.stderr.strip() or result.stdout.strip()}")
        return False

key = "3c6da7e6b3651be50123ab819d5e68c5"
usernames = ["jana107", "janarthanan107", "Janarthanan107", "janatheboss"]

success = False
for u in usernames:
    if try_auth(u, key):
        success = True
        break

if not success:
    # Try with KGAT prefix just in case but unlikely
    key_prefix = "KGAT_3c6da7e6b3651be50123ab819d5e68c5"
    if try_auth("jana107", key_prefix):
        success = True

if success:
    print("\nCorrect credentials found and configured!")
else:
    print("\nAll attempts failed.")
