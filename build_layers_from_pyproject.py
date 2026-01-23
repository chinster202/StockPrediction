import os
import subprocess
import shutil
import zipfile
import sys
import tomllib  # Built in since Python 3.11

# === CONFIG ===
PYTHON_VERSION = "3.14"              # Match your Lambda runtime
PYPROJECT_FILE = "pyproject.toml"
MAX_LAYER_SIZE_MB = 240              # Keep below AWS 250 MB limit
OUTPUT_DIR = "layers"
SKIP_PACKAGES = {"ruff", "black", "mypy", "ipython", "notebook"}

# Packages known to be too big for Lambda layers
LARGE_PACKAGES = {
    "torch": "≈1–2 GB — use a custom Lambda container instead",
    "tensorflow": "≈2 GB — container required",
    "torchvision": "hundreds MB — likely too large",
}


# === HELPERS ===
def get_dir_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 * 1024)


def create_layer_dir(index):
    path = os.path.join(
        OUTPUT_DIR, f"layer{index}", f"python/lib/python{PYTHON_VERSION}/site-packages"
    )
    os.makedirs(path, exist_ok=True)
    return path


def zip_layer(layer_folder):
    zip_path = os.path.join(OUTPUT_DIR, f"{os.path.basename(layer_folder)}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(layer_folder):
            for f in files:
                full_path = os.path.join(root, f)
                arcname = os.path.relpath(full_path, layer_folder)
                zf.write(full_path, arcname)
    print(f"✅ Zipped {zip_path}")


def load_dependencies(pyproject_path):
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    try:
        deps = data["project"]["dependencies"]
    except KeyError:
        print("❌ No [project.dependencies] found in pyproject.toml")
        sys.exit(1)
    return deps


# === MAIN ===
def main():
    if not os.path.exists(PYPROJECT_FILE):
        print(f"❌ {PYPROJECT_FILE} not found")
        sys.exit(1)

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dependencies = load_dependencies(PYPROJECT_FILE)
    print(f"📦 Found {len(dependencies)} total dependencies")

    # Filter out dev tools
    filtered = []
    for dep in dependencies:
        pkg_name = dep.split(">=")[0].split("==")[0].strip().lower()
        if pkg_name in SKIP_PACKAGES:
            print(f"🧹 Skipping dev tool: {pkg_name}")
            continue
        if pkg_name in LARGE_PACKAGES:
            print(f"⚠️  {pkg_name} is very large. {LARGE_PACKAGES[pkg_name]}")
            continue
        filtered.append(dep)

    if not filtered:
        print("No deployable dependencies left after filtering.")
        sys.exit(0)

    print(f"➡️  Installing {len(filtered)} production dependencies...\n")

    layer_index = 1
    current_layer_path = create_layer_dir(layer_index)

    for pkg in filtered:
        print(f"Installing {pkg} ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--target", current_layer_path, pkg],
            check=True,
        )

        size_mb = get_dir_size_mb(os.path.join(OUTPUT_DIR, f"layer{layer_index}"))
        if size_mb > MAX_LAYER_SIZE_MB:
            print(f"⚠️ layer{layer_index} exceeded {MAX_LAYER_SIZE_MB} MB. Moving {pkg} to a new layer.")
            shutil.rmtree(current_layer_path)
            layer_index += 1
            current_layer_path = create_layer_dir(layer_index)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--target", current_layer_path, pkg],
                check=True,
            )

    # Zip each layer
    for i in range(1, layer_index + 1):
        folder = os.path.join(OUTPUT_DIR, f"layer{i}")
        zip_layer(folder)

    print("\n🎉 Done! Upload each .zip in the 'layers/' folder as AWS Lambda Layers.")
    print("💡 Tip: You can attach up to 5 layers per function.")

if __name__ == "__main__":
    main()
