THIS_FILE := $(lastword $(MAKEFILE_LIST))
# Project settings
PACKAGE := $(firstword $(shell poetry version))
PROJECT := PACKAGE
VERSION = $(word 2,$(shell poetry version))

# Project paths
CONFIG := $(wildcard *.py)
SOURCES := $(shell find ./$(PACKAGE) -name "*.py")
# Virtual environment paths
VIRTUAL_ENV := .venv

# ARG processing
DEFAULTARG=patch

# aux functions
include makeutils/functions.mk

# commands
BUMPVERSION= $(VIRTUAL_ENV)/bin/bump2version
RUN := poetry run

# MAIN TASKS ##################################################################

.DEFAULT: ;

.PHONY: all
all: install

.PHONY: ci
ci: format check test ## Run all tasks that determine CI status

.PHONY: run ## Start the program
run: install
	$(RUN) python $(PACKAGE)/__main__.py

# SYSTEM DEPENDENCIES #########################################################

.PHONY: doctor
doctor:  ## Confirm system dependencies are available
	bin/verchew

# PROJECT DEPENDENCIES ########################################################

DEPENDENCIES := $(VIRTUAL_ENV)/.poetry-$(shell bin/checksum pyproject.toml poetry.lock)

.PHONY: install
install: $(DEPENDENCIES) .cache
	@if ! $(call check-package,$(PACKAGE)) ; then      \
	  echo "$(PACKAGE) not installed. Installing ..." ;\
	  poetry install                                  ;\
	fi

$(DEPENDENCIES): poetry.lock
	@ poetry config virtualenvs.in-project true
	poetry install --no-root
	@if $(call check-package,pre-commit) ; then      \
	  $(RUN) pre-commit install                     ;\
	fi
	@ touch $@

poetry.lock:
	poetry lock
	@ touch $@

.cache:
	@ mkdir -p .cache

# CHECKS ######################################################################


.PHONY: check
check: install format  ## Run formaters, linters, and static analysis
ifdef CI
	git diff --exit-code
endif
	$(RUN) pydocstyle $(PACKAGE) $(CONFIG)

# TESTS #######################################################################

RANDOM_SEED ?= $(shell date +%s)
FAILURES := .cache/v/cache/lastfailed

PYTEST_OPTIONS := --random --random-seed=$(RANDOM_SEED)
ifdef DISABLE_COVERAGE
PYTEST_OPTIONS += --no-cov --disable-warnings
endif
PYTEST_RERUN_OPTIONS := --last-failed --exitfirst

.PHONY: test-prod
test-prod: RUN =
test-prod: test-all

.PHONY: test
test: test-all ## Run unit and integration tests

.PHONY: test-unit
test-unit: install
	@ ( mv $(FAILURES) $(FAILURES).bak || true ) > /dev/null 2>&1
	$(RUN) pytest $(PACKAGE) $(PYTEST_OPTIONS)
	@ ( mv $(FAILURES).bak $(FAILURES) || true ) > /dev/null 2>&1

.PHONY: test-int
test-int: install
	@ if test -e $(FAILURES); then $(RUN) pytest tests $(PYTEST_RERUN_OPTIONS); fi
	@ rm -rf $(FAILURES)
	$(RUN) pytest tests $(PYTEST_OPTIONS)

.PHONY: test-all
test-all: install
	@ if test -e $(FAILURES); then $(RUN) pytest $(PACKAGE) $(PYTEST_RERUN_OPTIONS); fi
	@ rm -rf $(FAILURES)
	$(RUN) pytest $(PACKAGE) tests $(PYTEST_OPTIONS)

.PHONY: read-coverage
read-coverage:
	bin/open htmlcov/index.html

# DOCUMENTATION ###############################################################

SPHINX_INDEX := docs/_build/index.html

.PHONY: docs
docs: sphinx uml ## Generate documentation and UML

.PHONY: sphinx
sphinx: install $(SPHINX_INDEX)

$(SPHINX_INDEX): docs/*.md
	@ mkdir -p docs/about
	@ cd docs && ln -sf ../README.md README.md
	@ cd docs/about && ln -sf ../../CHANGELOG.md changelog.md
	@ cd docs/about && ln -sf ../../CONTRIBUTING.md contributing.md
	@ cd docs/about && ln -sf ../../LICENSE.md license.md
	cd docs && $(MAKE) clean
	cd docs && $(MAKE) html

.PHONY: uml
uml: install docs/*.png
docs/*.png: $(SOURCES)
	$(RUN) pyreverse $(PACKAGE) -p $(PACKAGE) -a 1 -f ALL -o png --ignore tests
	- mv -f classes_$(PACKAGE).png docs/classes.png
	- mv -f packages_$(PACKAGE).png docs/packages.png

# BUILD #######################################################################

EXE_FILES := dist/$(PACKAGE).*

SDIST := dist/$(PACKAGE)-$(VERSION)*.tar.gz
WHEEL := dist/$(PACKAGE)-$(VERSION)*.whl

DIST_FILES := $(SDIST) $(WHEEL)

.PHONY: dist
dist: install $(DIST_FILES)
	@$(call inf, Copying setup.py to root directory...)
	@ tar -zx --strip-components=1 -f $(SDIST) $(PACKAGE)-$(VERSION)/setup.py

$(SDIST): $(SOURCES) pyproject.toml
	poetry build -f sdist

$(WHEEL): $(SOURCES) pyproject.toml
	@ poetry build -f wheel

# RELEASE #####################################################################

.PHONY: version version-chk print-version
version-chk:
	@$(call check-create-branch,master)
	@$(call assert-command-present,$(BUMPVERSION))
	@$(call check-file-changes,poetry.lock)
	@$(call check-file-changes,setup.py)
	@$(call check-wd)
	@$(call check-upstream,master)

version: install version-chk ## Make a version bump and push it to master
	@ printf "Checking out master ...\n" && git checkout master --quiet
	@ $(RUN) $(BUMPVERSION) $(FIRSTARG)
	@ git push -o ci.skip origin master
	@ git push origin master --tags
	@ $(MAKE) --no-print-directory -f $(THIS_FILE) print-version

print-version:
	@ $(call inf,version dump: $(VERSION) OK!)

.PHONY: release
release: version

# .PHONY: upload
# upload: dist ## Upload the current version to pypiserver
#	@ poetry publish -r github


# CLEANUP #####################################################################

.PHONY: clean
clean: .clean-build .clean-docs .clean-test .clean-install ## Delete all generated and temporary files

.PHONY: clean-all
clean-all: clean
	rm -rf $(VIRTUAL_ENV)

.PHONY: .clean-install
.clean-install:
	find $(PACKAGE) -path '*/__pycache__*' -delete
	find $(PACKAGE) -type d -name '__pycache__' -empty -delete
	rm -rf *.egg-info

.PHONY: .clean-test
.clean-test:
	rm -rf .cache .pytest .coverage htmlcov

.PHONY: .clean-docs
.clean-docs:
	rm -rf docs/*.png site

.PHONY: .clean-build
.clean-build:
	rm -rf *.spec dist build

# HELP ########################################################################

.PHONY: help
help: all
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(firstword $(MAKEFILE_LIST)) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

.PHONY: test-makefile-functions
test-makefile-functions:
	@$(call inf,¡test info!)
	@$(call warn,¡test warning!)
	@$(call err,¡test error!)
	@$(call assert-command-present,python)

.PHONY: variables
variables:
	@$(call inf, SOURCES:= $(SOURCES))
