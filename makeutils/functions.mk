not-empty = $(if $(strip $1),T)
empty = $(if $(strip $1),,T)

PIP := $(VIRTUAL_ENV)/bin/pip
which = which $1 >/dev/null || which $(VIRTUAL_ENV)/bin/$1 >/dev/null
RED=\e[0;31m
LRED=\e[1;31m
LCYAN=\e[96m
LYEL=\e[93m
NC=\e[0m

ARGS = $(filter-out $(word 1,$(MAKECMDGOALS)), $(MAKECMDGOALS))
FIRSTARG = $(if $(call empty, $(ARGS)),$(DEFAULTARG),$(word 1,$(ARGS)))

define err
printf "$(LRED)%s$(NC)\n" "$(1)"
endef

define warn
printf "$(LYEL)%s$(NC)\n" "$(1)"
endef

define inf
printf "$(LCYAN)%s$(NC)\n" "$(1)"
endef

# check-package(package)
#   Checks if package is installed within venv
define check-package
$(PIP) show --quiet $1
endef

# check-file-changes(file)
#   Checks if file must be committed and commits it if necessary
define check-file-changes
printf "Commiting %s if required ..." "$(1)"
if git diff --name-only | grep -q $1; then             \
  $(call inf,yes)                                    ; \
  git add $1                                         ; \
	git commit --quiet -m"$1"                          ; \
else                                                   \
  $(call inf,no)                                     ; \
fi
endef

# check-wd
#   First checks if working directory is clean and then, if staging area is
#   also clean.
#   If any condition fails, prints error message and returns 1
define check-wd
printf "Checking working directory..."
if git diff --quiet --exit-code ; then                 \
  $(call inf,ok)                                     ; \
else                                                   \
  $(call err,working directory not clean. Aborting)  ; \
  printf "Changes:"                                    ; \
  git diff --name-only                               ; \
  exit 1                                             ; \
fi
printf "Checking staging area..."
if git diff --cached --quiet --exit-code ; then        \
  $(call inf,ok)                                     ; \
else                                                   \
  $(call err,staging area not clean. Aborting)       ; \
  printf "Changes:"                                    ; \
  git diff --cached --name-only                      ; \
  exit 1                                             ; \
fi
endef

# assert-command-present(command)
#   Checks if a command exists within the PATH or the python enviroment defined
#   in VIRTUAL_ENV. If not, prints an error message and returns 1
define assert-command-present
printf "Checking for %s..." "$(1)"
if $(call which,$1) ; then                              \
  $(call inf,ok)                                      ; \
else                                                    \
  $(call err,'$1' missing and needed for this build)  ; \
  exit 1                                              ; \
fi
endef

# check-create-branch(branch)
#   Checks the existance of a branch in the local repo.
#   If do not exist, creates the branch
define check-create-branch
printf "Checking if %s exists..." "$(1)"
if git rev-parse --verify --quiet $1 >/dev/null; then           \
  $(call inf,ok)                                              ; \
else                                                            \
  $(call inf,branch $1 does not exist. Creating branch $1...) ; \
	git branch $1                                             ; \
fi
endef

# check-upstream(branch)
#   Checks if there are changes in the remote branch. For that, updates the
#   remote tracking branches and compares them with local HEAD. If the remote
#   is ahead, prints an error message and returns 1
define check-upstream
if ! git branch -r | grep -q origin/$1 ; then                         \
  $(call warn,origin/$1 does not exist)                             ; \
  printf "Checking out %s ..." "$(1)" && git checkout $1 --quiet             ; \
  printf "Creating origin/%s..." "$(1)" && git push --set-upstream origin $1 ; \
fi
printf "Checking if there are any incoming changes on %s..." "$(1)"
git fetch --quiet >/dev/null
log=$$(git log $1..origin/$1 --oneline | wc -w)                     ; \
if [ $$log -eq 0 ]; then                                              \
  $(call inf,ok)                                                    ; \
else                                                                  \
  $(call warn,the branch $1 is behind origin/$1. You need to merge) ; \
  exit 1                                                            ; \
fi
endef
