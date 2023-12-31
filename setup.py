#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="adamantium",
    version="1.0.0",
    description="EMLOv3 Base Setup",
    author="Salil Gautam",
    author_email="salil.gtm@gmail.com",
    url="https://github.com/salil-gtm/emlov3_assignment_8",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "adamantium_train = adamantium.train:main",
            "adamantium_eval = adamantium.eval:main",
            "adamantium_infer = adamantium.infer:main",
            "adamantium_demo_cifar10 = adamantium.demo.cifar10_jit.gradio:main",
            "adamantium_demo_gpt = adamantium.demo.gpt_jit.gradio:main"
        ]
    },
)
