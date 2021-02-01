# cuda-mat

Contains some matrix transformation functions writed on cuda.


## Why not opencv::cuda?

`opencv::cuda` has simple and fast implementation, but there are several problems:

1. Not thread safe - some transformation functions from `opencv::cuda` uses gpu
constant memory, so you can't use same function in other thread without blocking

2. There are problem with signed integer overflow, see
[this issue](https://github.com/opencv/opencv_contrib/issues/2361)


## TODO

1. Add algorithm for inversing 3x3 transformation matricies and remove opencv
dependency
