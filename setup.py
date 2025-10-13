from setuptools import setup, find_packages
from typing import List

def get_requirements() -> List[str]:
    """Read dependencies from requirements.txt, ignoring empty lines and '-e .'."""
    requirements_list = []
    try:
        with open('requirements.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != '-e .':
                    requirements_list.append(requirement)
    except FileNotFoundError:
        # You can log this instead of printing in production
        print("requirements.txt file not found.")
    return requirements_list


print(get_requirements())


setup(
    name='Network_Security',
    version='0.0.1',
    author='Aviroop Ghosh',
    author_email='ghoshaviroop542@example.com',
    packages=find_packages(),
    install_requires=get_requirements(),
)