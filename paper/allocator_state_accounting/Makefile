TECTONIC ?= tectonic
PAPER_TEX := allocator_state_accounting.tex
PAPER_PDF := allocator_state_accounting.pdf
CACHE_DIR := .tectonic-cache
CONFIG_DIR := .tectonic-config
REPO_ROOT := $(abspath $(CURDIR)/../..)
LLM_OPTIMIZATIONS_ROOT ?= $(abspath $(REPO_ROOT)/../llm-optimizations)
LLM_OPTIMIZATIONS_PDF_DIR ?= $(LLM_OPTIMIZATIONS_ROOT)/docs/papers
LLM_OPTIMIZATIONS_PDF_NAME ?= allocator-runtime-core-plugin__allocator_state_accounting.pdf

.DEFAULT_GOAL := help

.PHONY: help pdf clean

help:
	@printf '%s\n' \
		'Paper targets:' \
		'  make pdf    Build the placeholder paper scaffold with tectonic' \
		'  make clean  Remove local LaTeX build outputs'

pdf:
	mkdir -p $(CACHE_DIR) $(CONFIG_DIR)
	XDG_CACHE_HOME=$(CURDIR)/$(CACHE_DIR) XDG_CONFIG_HOME=$(CURDIR)/$(CONFIG_DIR) $(TECTONIC) --keep-logs --keep-intermediates $(PAPER_TEX)
	@if [ -d "$(LLM_OPTIMIZATIONS_ROOT)" ] && [ -f "$(PAPER_PDF)" ]; then \
		python3 "$(LLM_OPTIMIZATIONS_ROOT)/scripts/export_paper_pdf.py" \
			--workspace-file "$(LLM_OPTIMIZATIONS_ROOT)/llm-optimizations.code-workspace" \
			--repo-root "$(REPO_ROOT)" \
			--source-pdf "$(PAPER_PDF)" \
			--output-name "$(LLM_OPTIMIZATIONS_PDF_NAME)" \
			--destination-dir "$(LLM_OPTIMIZATIONS_PDF_DIR)"; \
	fi

clean:
	rm -f *.aux *.log *.out *.toc *.blg *.bbl $(PAPER_PDF)
	rm -rf $(CACHE_DIR) $(CONFIG_DIR)
