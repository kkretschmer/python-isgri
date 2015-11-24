from setuptools import setup, find_packages

setup(
    name = 'integral_isgri',
    version = '0.1',
    description = 'INTEGRAL/ISGRI with Python',
    url = 'https://github.com/kkretschmer/python-isgri',
    author = 'Karsten Kretschmer',
    author_email = 'kkretsch@apc.univ-paris7.fr',
    license = 'GPL2+',
    package_dir = {'': 'src'},
    packages = find_packages('src'),
    entry_points = {
        'console_scripts': [
            'isgri-bgchi2 = integral_isgri.bgchi2:main',
            'isgri-bgcube-httpd = integral_isgri.bgserver:serve_cubes',
            'isgri-bglincomb-mktemplate = integral_isgri.bglincomb:mktemplate',
            'isgri-bglincomb-mkcube = integral_isgri.bglincomb:mkcube',
            'isgri-stack-cubes = integral_isgri.cubestack:stack_cubes',
            'isgri-stack2osabkg = integral_isgri.bgcube:stack2osa'
        ]
    },
    install_requires = [
        'astropy',
        'fitsio',
        'future',
        'numpy',
        'progressbar2',
        'scipy',
        'six'
    ]
)
