# simple makefile to simplify repetitive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
PYTESTS ?= pytest
CODESPELL_SKIPS ?= "docs/auto_*,*.fif,*.eve,*.gz,*.tgz,*.zip,*.mat,*.stc,*.label,*.w,*.bz2,*.annot,*.sulc,*.log,*.local-copy,*.orig_avg,*.inflated_avg,*.gii,*.pyc,*.doctree,*.pickle,*.inv,*.png,*.edf,*.touch,*.thickness,*.nofix,*.volume,*.defect_borders,*.mgh,lh.*,rh.*,COR-*,FreeSurferColorLUT.txt,*.examples,.xdebug_mris_calc,bad.segments,BadChannels,*.hist,empty_file,*.orig,*.js,*.map,*.ipynb,searchindex.dat,install_mne_c.rst,plot_*.rst,*.rst.txt,c_EULA.rst*,*.html,gdf_encodes.txt,*.svg"
CODESPELL_DIRS ?= morf/ docs/ examples/ tests/

all: clean inplace test

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	$(PYTHON) setup.py clean
	rm -rf _build
	rm -rf _build
	rm -rf dist
	rm -rf morf.egg-info

clean-ctags:
	rm -f tags
	rm junit-results.xml

clean-ctags:
	rm -f tags

clean: clean-build clean-so clean-ctags

inplace:
	$(PYTHON) setup.py develop

test-coverage:
	rm -rf coverage .coverage
	$(PYTESTS) --cov=morf --cov-report html:coverage


check-manifest:
	check-manifest --ignore .circleci/*,docs,.DS_Store

reqs:
	pipfile2req --dev > test_requirements.txt
	pipfile2req > requirements.txt
	pipfile2req > docs/requirements.txt
	pipfile2req --dev >> docs/requirements.txt

flake:
	@if command -v flake8 > /dev/null; then \
		echo "Running flake8"; \
		flake8 --count morf examples tests; \
	else \
		echo "flake8 not found, please install it!"; \
		exit 1; \
	fi;
	@echo "flake8 passed"

pydocstyle:
	@echo "Running pydocstyle"
	@pydocstyle

codespell:  # running manually
	@codespell -w -i 3 -q 3 -S $(CODESPELL_SKIPS) --ignore-words=ignore_words.txt $(CODESPELL_DIRS)

codespell-error:  # running on travis
	@echo "Running code-spell check"
	@codespell -i 0 -q 7 -S $(CODESPELL_SKIPS) --ignore-words=ignore_words.txt $(CODESPELL_DIRS)

type-check:
	mypy ./morf

pep:
	@$(MAKE) -k flake pydocstyle check-manifest codespell-error type-check

build-dev:
	pip install --verbose --no-build-isolation --editable .

build-doc:
	cd docs; make clean
	cd docs; make html

build-pipy:
	python setup.py sdist bdist_wheel

test-pipy:
	twine check dist/*
	twine upload --repository testpypi dist/*

upload-pipy:
	twine upload dist/*