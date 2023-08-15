#!/usr/bin/env bash

mkdir data

gdown "https://drive.google.com/uc?export=download&id=1xCGELdqS5dYQKeUltHS_uRQ7NQRfKJez"
unzip rosbank-ml-contest-boosters.pro.zip -d data
mv rosbank-ml-contest-boosters.pro.zip data/
