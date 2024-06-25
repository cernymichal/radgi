# radgi

CUDA accelerated global illumination baker based on the Radiosity method, with a lot of room for improvement.

Specifically it uses ray-casting for form factor determination between pathes that are made for every texel in the desired lightmap. Scenes are loaded from an OBJ file with their respective materials. (albedo and emission) The scene polygons are rasterized to the lightmap, a BVH is built, and then the calculation can start. There are 3 methods implemented: CPU progressive, CPU gathering and GPU (CUDA) gathering. While progressive is single threaded and biased towards examining the patch with most unshot energy, the gathering approach simulates whole light bounces on many threads.

This project was made for the NI-GPU course at FIT CTU during the summer semester of 2023/24, the report (in czech) is available [here](https://media.githubusercontent.com/media/cernymichal/radgi/master/report.pdf).

|  |  |  |
|:---:|:---:|:---:|
| ![](./resources/cornell_box/output.png) | ![](./resources/cornell_box/output_interpolated.png) | ![](./resources/cornell_box/reference.png) |
| radgi 64x64 lightmap | radgi 64x64 lightmap (interpolated) | path-traced reference |

## Building

- CMake >3.28
- G++ 13
- CUDA Toolkit 12.5

```sh
cmake --preset=linux-x64-release
```

```sh
cmake --build --preset=linux-x64-release -j 12
```
