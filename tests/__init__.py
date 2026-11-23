import os

# Keep the developer's own ~/.config/agent-cli/skills out of the tests.
os.environ["AGENT_GLOBAL_SKILLS_DIR"] = ""
