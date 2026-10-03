from setuptools import setup,Extension
setup(name='iev2mol-runtime',version='0.1.0',packages=['iev_math'],package_dir={'iev_math':'MAIN/model/iev_math'},ext_modules=[Extension('iev_math._legacy_math',['MAIN/model/iev_math/_legacy_math.c'],extra_compile_args=['-O2','-fno-fast-math','-ffp-contract=off'])],python_requires='>=3.12,<3.13')
