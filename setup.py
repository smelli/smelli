from setuptools import setup, find_packages


with open('smelli/_version.py') as f:
    exec(f.read())


with open('README.md', encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()


setup(name='smelli',
      version=__version__,
      author='Jason Aebischer <jason.aebischer@tum.de>, Jacky Kumar <jacky.kumar@umontreal.ca>, Peter Stangl <peter.stangl@lapth.cnrs.fr>, David M. Straub <david.straub@tum.de>',
      description='A Python package providing a global likelihood function in the space of dimension-6 Wilson coefficients of the Standard Model Effective Field Theory (SMEFT)',
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      license='MIT',
      packages=find_packages(),
      package_data={
        'smelli': ['data/yaml/*.yaml',
                   'data/cache/*.p',
                   'data/test/*.yaml',
                   ]
      },
      install_requires=[
        'numpy>=1.16.5',
        'flavio>=' + __flavio__version__,
        'wilson',
        'pandas',
        'multipledispatch'
      ],
      extras_require={
            'testing': ['nose'],
      },
      )
