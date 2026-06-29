from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "xqcpp",
        ["xqcpp.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["/O2", "/std:c++17", "/utf-8"],
    ),
]

setup(
    name="xqcpp",
    version="0.1",
    ext_modules=ext_modules,
)