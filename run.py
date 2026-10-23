"""
1-Click Launcher for AI-Based Cataract Clinical Decision Support System (CDSS)
"""

import sys
import os
import subprocess
import webbrowser
import time

def main():
    print("=" * 70)
    print("   AI-BASED CATARACT CLINICAL DECISION SUPPORT SYSTEM (CDSS)")
    print("=" * 70)
    
    # Automatically switch to 64-bit Miniconda Python on E: drive for full AI deep learning
    e_conda_py = r"E:\miniconda3\python.exe"
    if sys.executable.lower() != e_conda_py.lower() and os.path.exists(e_conda_py):
        print("[*] Switching to 64-bit Miniconda on E: drive for AI deep learning acceleration...")
        os.execv(e_conda_py, [e_conda_py] + sys.argv)
            
    print("\n[*] Checking environment & dependencies...")
    
    # Ensure dependencies are available
    required_packages = ["fastapi", "uvicorn", "PIL", "numpy", "multipart"]
    missing = []
    
    for pkg in ["fastapi", "uvicorn", "PIL", "numpy"]:
        try:
            if pkg == "PIL":
                import PIL
            else:
                __import__(pkg)
        except ImportError:
            missing.append(pkg)
            
    if missing:
        print(f"[!] Installing required dependencies: {missing}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn", "pillow", "numpy", "python-multipart"])
        print("[OK] Dependencies installed successfully!")
    else:
        print("[OK] All essential packages are ready.")
        
    # Check if server is already active on port 8000
    try:
        import urllib.request
        urllib.request.urlopen("http://127.0.0.1:8000/api/patients", timeout=1)
        print("[*] Cataract CDSS Server is already active on http://127.0.0.1:8000!")
        print("[*] Opening Doctor Decision Support Dashboard in your browser...\n")
        webbrowser.open("http://127.0.0.1:8000")
        return
    except Exception:
        pass

    print("\n[*] Starting Cataract CDSS Server on http://127.0.0.1:8000 ...")
    print("[*] Loading AI models & opening Doctor Decision Support Dashboard...\n")
    
    # Open browser only after the AI backend server is fully initialized and responding
    def wait_and_open_browser():
        import urllib.request
        for _ in range(60):
            try:
                urllib.request.urlopen("http://127.0.0.1:8000/api/patients", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        webbrowser.open("http://127.0.0.1:8000")
        
    import threading
    threading.Thread(target=wait_and_open_browser, daemon=True).start()
    
    import uvicorn
    # Change working directory to project root
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()
