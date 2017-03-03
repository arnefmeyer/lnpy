# simple makefile to simplify repetetive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
CYTHON ?= cython
CTAGS ?= ctags

all: clean inplace

clean-ctags:
	rm -f tags

clean: clean-ctags
	$(PYTHON) setup.py clean
	rm -rf dist

inplace:
	$(PYTHON) setup.py build_ext -i

trailing-spaces:
	find lnpy -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

cython:
	$(CYTHON) --cplus lnpy/learn/pyhelper.pyx
	$(CYTHON) --cplus lnpy/lnp/fast_tools.pyx
	$(CYTHON) lnpy/transform/wrap_gtfb.pyx
	$(CYTHON) lnpy/multilinear/context/context_fast.pyx
	
doc: inplace
	$(MAKE) -C doc html

