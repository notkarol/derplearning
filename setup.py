from setuptools import setup, find_packages

setup(name='derp',
      version='0.1',
      description='Race your RC car autonomously',
      long_description=__doc__,
      license='MIT',
      tests_require=["pytest"],
      include_package_data=True,
      zip_safe=False,
      packages=['derp', 'derp.controllers', 'derp.components', 'derp.models'],
      install_requires=find_packages()
)
