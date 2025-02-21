# list all available commands
default:
  just --list

###############################################################################
# Basic project and env management

# clean all build, python, and lint files
clean:
	rm -fr dist
	rm -fr .eggs
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -fr .mypy_cache
	rm -fr .pytest_cache
	rm -fr .ruff_cache

# install with all deps
install-pipeline-deps:
	pip install uv
	uv pip install -r requirements-pipeline.txt

# setup psl environment
setup-psl-env:
    conda env create -f psl-environment.yml -y

# lint, format, and check all files
lint:
    pip install uv
    uv pip install precommit
    pre-commit run --all-files