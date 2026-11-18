"""Build RL_LAB4_<StudentId>_<Name>.zip from src/ (code only) and report.pdf."""
import argparse
import fnmatch
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXCLUDED_DIRS = {"runs", "videos", "__pycache__", "log", ".venv"}
EXCLUDED_FILES = ["*.pt", "*.pth", "*.pyc"]


def ignore(directory, names):
	ignored = []
	for name in names:
		if name in EXCLUDED_DIRS or any(fnmatch.fnmatch(name, pattern) for pattern in EXCLUDED_FILES):
			ignored.append(name)
	return ignored


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--student_id", required=True)
	parser.add_argument("--name", required=True)
	args = parser.parse_args()

	package_name = f"RL_LAB4_{args.student_id}_{args.name}"
	output_dir = ROOT / "submission"
	package_dir = output_dir / package_name
	if package_dir.exists():
		shutil.rmtree(package_dir)

	shutil.copytree(ROOT / "src", package_dir / "src", ignore=ignore)

	report = ROOT / "report.pdf"
	if report.exists():
		shutil.copy2(report, package_dir / "report.pdf")
	else:
		print("WARNING: report.pdf not found in the project root, the zip does not contain it.")

	archive = shutil.make_archive(str(output_dir / package_name), "zip", output_dir, package_name)
	print(f"Created: {archive}")


if __name__ == "__main__":
	main()
