from setuptools import setup, find_packages
from uos_complex import __version__
#from pip.req import parse_requirements
try: # for pip >= 10
    from pip._internal.req import parse_requirements
    from pip._internal.download import PipSession
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements
    from pip.download import PipSession
################################################################################
# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements("requirements.txt", session=PipSession())
reqs = [str(ir.req) for ir in install_reqs]


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
