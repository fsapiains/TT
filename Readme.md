1) Instalar OpenCV
2) Instalar libboost (utilizada para iterar archivos dentro de las carpetas)
3) Para el módulo de detección de personas bajar de la base de datos de inria las imágenes de entrenamiento: ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar
4) Para compilar utilizando libboost: g++ -ggdb Archivo.cpp -o Archivo $(pkg-config --cflags --libs opencv) -lboost_filesystem -lboost_system 