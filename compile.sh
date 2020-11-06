#!/bin/bash




build_dir=build
[ -d $build_dir ] || mkdir $build_dir
cd $build_dir 
    cmake -G "Sublime Text 2 - Unix Makefiles" ../
    make -j
    mv -v *.so ../
cd ..


