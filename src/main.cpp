#include <argh.h>

#include "GISolver/CUDASolver/CUDASolver.h"
#include "GISolver/GatheringSolver.h"
#include "GISolver/ProgressiveSolver.h"
#include "Scene.h"

auto constexpr USAGE =
    R"(Usage: radgi <input_file> [-o output_file] [-r resolution] [-t threshold] [-i bounces] [-e radius]

Options:
    <input_file>                    Input scene in .obj format

	-o, --output <output_file>      Output file for the lightmap (output.exr by default)

	-r, --resolution <resolution>   Lightmap resolution (128 by default)

	-t, --threshold <threshold>     Residue threshold for terminating (0.1 by default)
                                    This is the default CPU mode - progressive shooting.
                                    The scene is solved until the maximum patch residue is below the threshold.

	-b, --bounces <bounces>         Number of gathering bounces through the whole scene
     			                    When set, -t is ignored and the scene is solved by gathering instead of shooting.

	-d, --dilation <radius>         Extrapolate lightmap islands with the given radius in pixels (2 by default)

    -g, --gpu                       Use CUDA acceleration (4 bounces by default)
                                    Only the gathering mode is supported on the GPU. -t is ignored.

    -h, --help                      Print this help message


Example:
	radgi resources/cornell_box/cornell_box.obj -o lightmap.exr -r 256 -t 0.01)";

int main(int argc, char* argv[]) {
    argh::parser cmdl(argc, argv, argh::parser::PREFER_PARAM_FOR_UNREG_OPTION);

    if (!cmdl(1) || cmdl[{"-h", "--help"}]) {
        LOG(USAGE);
        return EXIT_FAILURE;
    }
    auto inputFile = cmdl[1];

    std::string outputFile;
    cmdl({"-o", "--output"}, "output.exr") >> outputFile;

    u32 resolution;
    cmdl({"-r", "--resolution"}, 128) >> resolution;

    f32 threshold;
    cmdl({"-t", "--threshold"}, 0.1f) >> threshold;
    bool useShooting = true;

    bool useCUDA = cmdl[{"-g", "--gpu"}];

    u32 bounces = 4;
    if (cmdl({"-b", "--bounces"}) >> bounces || useCUDA)
        useShooting = false;

    u32 dilationRadius;
    cmdl({"-d", "--dilation"}, 2) >> dilationRadius;

    auto lightmapSize = uvec2(resolution);

    auto scene = makeRef<Scene>();
    scene->addMesh(loadMesh(inputFile));
    scene->initialize(lightmapSize);

// #define OUTPUT_PATCH_GEOMETRY
#ifdef OUTPUT_PATCH_GEOMETRY
    saveMesh("patches.obj", scene->createPatchGeometry());
#endif

    Ref<IGISolver> solver;
    if (useCUDA) {
        LOG("Using GPU mode");
        solver = makeRef<CUDASolver>(bounces);
    }
    else {
        LOG("Using CPU mode");
        if (useShooting)
            solver = makeRef<ProgressiveSolver>(threshold);
        else
            solver = makeRef<GatheringSolver>(bounces);
    }

    solver->initialize(scene);

    LOG("Solving");
    auto start = std::chrono::high_resolution_clock::now();
    auto lightmap = solver->solve();
    auto finish = std::chrono::high_resolution_clock::now();

    LOG(std::format("Solved in {:.2f}s", std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() / 1000.0));

    LOG("Dilating the lightmap");
    scene->dilateLightmap(lightmap, dilationRadius);

    lightmap.save(outputFile, true);

    return EXIT_SUCCESS;
}
