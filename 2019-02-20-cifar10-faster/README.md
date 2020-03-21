# CIFAR10 en moins de 20 épochs - 20/02/2019

### Contexte et [Stanford DAWN Benchmarks sur CIFAR 10](https://dawn.cs.stanford.edu/benchmark/index.html#cifar10-train-time)

### Record actuel et exploration du code

Actuellement, le "record" est d'entrainer un réseau (type resnet) jusqu'à 94% accuracy sur test en 79 seconds sur Nvidia V100 en 24 épochs: https://dawn.cs.stanford.edu/benchmark/index.html#cifar10-train-time

Voici le repo pour reproduire le résultat : https://github.com/davidcpage/cifar10-fast
et la série d'articles qui explique cela plus en détails : https://www.myrtle.ai/2018/09/24/how_to_train_your_resnet/

### Nouveau challenge : apprentissage en moins de 20 épochs. 

Et selon le blog ce résultat n'est pas encore la limite. Il paraît qu'on peut descendre au dessous de 20 épochs et réduire encore le temps. Le but de ce journal est de faire un zoom sur le résultat officiel, présenter mes petites expériences (qui actuellement donnent 92-93% entre 12-18 epochs sur Nvidia 1080Ti) et discuter comment améliorer la précision (accuracy) tout en gardant la vitesse d'apprentissage. J'ai crée un répo: https://github.com/vfdev-5/cifar10-faster pour les expérimentations.


