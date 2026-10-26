BUILD_DIR  ?= build
BUILD_TYPE ?= Release
CMAKE      ?= cmake
JOBS       ?= $(shell nproc 2>/dev/null || echo 2)
AUDIO_DIR  ?= audio
AUDIO      ?=
CAM        ?= 0

.DEFAULT_GOAL := help

.PHONY: all help configure build clean distclean rebuild run list-audio models probe

all: build

# Help
help:
	@echo "audiogesture build & run helper"
	@echo ""
	@echo "Usage:"
	@echo "  make                       Configure (if needed) and build"
	@echo "  make build                 Same as 'make'"
	@echo "  make clean                 Remove object files / executable"
	@echo "  make distclean             Remove whole $(BUILD_DIR)/ directory"
	@echo "  make rebuild               distclean + configure + build"
	@echo ""
	@echo "  make run                   Pick a .wav from $(AUDIO_DIR)/ and run"
	@echo "  make run AUDIO=foo.wav     Run a specific file (searched in ./ then $(AUDIO_DIR)/)"
	@echo "  make run CAM=1             Use camera index 1 (default 0)"
	@echo "  make list-audio            Show .wav files available in $(AUDIO_DIR)/"
	@echo "  make models                Download hand-tracking models into models/ (run once)"
	@echo "  make probe                 Build tools/hand_probe (test tracker on an image/video, no camera)"
	@echo ""
	@echo "Overridable vars:  BUILD_DIR=$(BUILD_DIR)  BUILD_TYPE=$(BUILD_TYPE)  AUDIO_DIR=$(AUDIO_DIR)  CAM=$(CAM)"
	@echo ""
	@echo "Note: 'make --help' prints GNU make's own help — use 'make help' for this menu."

# Build
configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build:
	@if [ ! -f $(BUILD_DIR)/CMakeCache.txt ]; then \
		$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) ; \
	fi
	$(CMAKE) --build $(BUILD_DIR) -j $(JOBS)

clean:
	@if [ -f $(BUILD_DIR)/CMakeCache.txt ]; then \
		$(CMAKE) --build $(BUILD_DIR) --target clean ; \
	else echo "$(BUILD_DIR)/ not configured — nothing to clean." ; fi

distclean:
	rm -rf $(BUILD_DIR)
	@echo "Removed $(BUILD_DIR)/"

rebuild: distclean
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	$(CMAKE) --build $(BUILD_DIR) -j $(JOBS)

# Hand-tracking models (MediaPipe palm + landmarks, ~8 MB) and OpenCV<4.7 compat patch
models:
	bash models/get_models.sh

probe: build
	$(CMAKE) --build $(BUILD_DIR) --target hand_probe -j $(JOBS)
	@echo "Built $(BUILD_DIR)/hand_probe  (usage: $(BUILD_DIR)/hand_probe photo.jpg --out result.png)"

# Audio listing
list-audio:
	@if [ -d "$(AUDIO_DIR)" ]; then \
	  files=$$(find "$(AUDIO_DIR)" -maxdepth 1 -type f -name '*.wav' | sort); \
	  if [ -z "$$files" ]; then echo "(no .wav files in $(AUDIO_DIR)/)" ; \
	  else echo "$(AUDIO_DIR)/ contents:"; echo "$$files" | sed 's/^/  /' ; fi ; \
	else echo "$(AUDIO_DIR)/ does not exist." ; fi

# Run with interactive picker
# Resolution order for audio file:
#   1. AUDIO=<path>   -> use literally, if readable
#   2. AUDIO=<name>   -> try ./<name>, then $(AUDIO_DIR)/<name>
#   3. unset          -> list $(AUDIO_DIR)/*.wav and prompt (needs a TTY)
#                        * 0 files  -> error
#                        * 1 file   -> auto-select
#                        * N files  -> numbered menu
run: build
	@bash -c '\
	set -e ; \
	audio="" ; \
	if [ -n "$(AUDIO)" ]; then \
	  if   [ -r "$(AUDIO)" ];              then audio="$(AUDIO)" ; \
	  elif [ -r "$(AUDIO_DIR)/$(AUDIO)" ]; then audio="$(AUDIO_DIR)/$(AUDIO)" ; \
	  else echo "ERROR: cannot find \''$(AUDIO)'\'' (tried ./, $(AUDIO_DIR)/)" >&2 ; exit 1 ; fi ; \
	else \
	  [ -d "$(AUDIO_DIR)" ] || { echo "ERROR: directory $(AUDIO_DIR)/ does not exist." >&2 ; exit 1 ; } ; \
	  mapfile -d "" files < <(find "$(AUDIO_DIR)" -maxdepth 1 -type f -name "*.wav" -print0 | sort -z) ; \
	  n=$${#files[@]} ; \
	  if   [ $$n -eq 0 ]; then echo "ERROR: no .wav files in $(AUDIO_DIR)/." >&2 ; exit 1 ; \
	  elif [ $$n -eq 1 ]; then audio=$${files[0]} ; echo "Only one file: $$audio" ; \
	  else \
	    [ -t 0 ] || { echo "ERROR: multiple .wav files; pass AUDIO=<name>.wav" >&2 ; exit 1 ; } ; \
	    PS3="Choose audio file (0 to cancel): " ; \
	    select f in "$${files[@]}" ; do \
	      if [ -n "$$f" ]; then audio=$$f ; break ; else echo "Cancelled." ; exit 1 ; fi ; \
	    done ; \
	  fi ; \
	fi ; \
	echo "Playing: $$audio  (camera $(CAM))" ; \
	exec ./$(BUILD_DIR)/audiogesture "$$audio" $(CAM)'
