import os

uri = os.environ.get("MODEL_URI", "undefined")
print(f"✅ Pretend validating model at: {uri}")
exit(0)
