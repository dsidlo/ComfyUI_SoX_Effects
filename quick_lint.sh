#!/usr/bin/env bash

uv sync --dev

echo "----------------------------------"
echo "py_compile src/*.py ---------------"
echo "----------------------------------"
echo
uv run python src/*.py || true
echo "----------------------------------"
echo "flake8 src/*.py ------------------"
echo "----------------------------------"
flake8 src/*.py --ignore E501,W291,W293,E301 || true

echo
echo "----------------------------------"
echo "text/*.py ------------------------";
echo "----------------------------------"
echo
uv run python tests/*.py || true
echo "----------------------------------"
echo "flake8 src/*.py ------------------"
echo "----------------------------------"
flake8 tests/*.py --ignore E501,W291,W293,E301 || true

