#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="vap",
    version="1.0.0",
    description="Voice Activity Projection: Self-Supervised Learning of Turn-taking Events",
    author="erikekst",
    author_email="erikekst@kth.se",
    url="https://github.com/ErikEkstedt/VoiceActivityProjection",
    packages=find_packages(include=["vap"]),
)
