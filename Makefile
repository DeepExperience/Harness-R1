.PHONY: check test

check:
	python scripts/check_release.py

test:
	python -m unittest discover -s tests -v
