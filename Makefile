.PHONY: build dist redist install install-from-source clean uninstall deploy annotate develop

build:
	python setup.py build

dist:
	$(RM) -r dist
	python setup.py sdist bdist_wheel

redist: clean dist

install:
	pip install . --use-feature=in-tree-build

install-from-source: dist
	pip install dist/*.tar.gz

clean:
	$(RM) -r build dist *.egg-info
	$(RM) -r build dist *.so
	$(RM) -r cython/*.cpp
	find . -name __pycache__ -exec rm -r {} +
	#git clean -fdX

uninstall:
	pip uninstall hyperclip

deploy:
	twine upload dist/*.tar.gz

annotate:
	python setup_annotate.py build_ext --inplace --force
	
develop:
	python setup.py develop