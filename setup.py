# Copyright (C) 2020 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

from setuptools import setup, find_packages
PACKAGE_VERSION= '1.0.0'
import sys
import os



for idx, argv in enumerate(sys.argv):
	if argv=='--prefix':
		INSTALL_PREFIX_PATH=sys.argv[idx+1]
		

		INSTALL_PREFIX_PATH = INSTALL_PREFIX_PATH + "/lib/python3.7/site-packages/"
		print ('INSTALL_PREFIX_PATH: {}'.format(INSTALL_PREFIX_PATH ))

		if not os.path.exists(INSTALL_PREFIX_PATH):
			os.makedirs(INSTALL_PREFIX_PATH)

		break


setup(
    name='human_action_intetion_recognition',
    version=PACKAGE_VERSION,
    author='Kourosh Darvish',
    author_email='kourosh.darvish@iit.it',
    packages = find_packages(where='../'),
    package_dir={
        '':'../',
    },
#    package_data = {'': ['*.txt']},
#    description='a packge to predict human actions and motions',
#    classifiers=[
#        "Programming Language :: Python :: 3",
#        "License :: OSI Approved :: GNU Lesser General Public License v2.1 or any later version",
#        "Operating System :: OS Independent",
#    	],
	python_requires='==3.7',
    entry_points={
        "console_scripts": [
            "human_action_intetion_recognition = scripts.__main__:main"
        ]
    },
#    install_requires=["human_action_intetion_recognition"],
)
