The project has been laid out with AI assistance in mind. I will use the Windsurf IDE + MCP servers to add context and manage:
1. Code base
2. Unit Tests
3. Obsidian Vault

## Configure Project
1. Assert "uv" is installed for package management
2. Navigate to the project root to install a virtual environment
	1. The project root is prefixed with "rxx_[Project Name]", where "r" means project root and "xx" is a two-digit code for lexicographic positioning in a sorted list (e.g. file viewer)

### Create virtual environment

```bash
ubuntu> cd /home/tj/02_Windsurf_Projects/r03_Gimbal_Angle_Root/
ubuntu> uv venv ./venv
ubuntu> source venv/bin/activate

venv> uv pip install -r requirements.txt
```

![[Pasted image 20250527102509.png]]