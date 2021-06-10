from setuptools import setup, find_packages
from uos_complex import __version__

################################################################################
# parse_requirements() returns generator of pip.req.InstallRequirement objects
with open("requirements.txt") as f:
	reqs = f.read().split()


setup(name='uos_complex',
		version = __version__,
		description='Python library for complex system and network science lab of University of Seoul.',
		author='Jung-Hoon Jung',
		author_email='jh.jung@uos.ac.kr',
		packages=find_packages(),      
		include_package_data=True,      # include files in MANIFEST.in
		python_requires = '>=3',
		install_requires=reqs
	 )
